from abc import ABC, abstractmethod
import re

from llm import call_openai


class MAPGDPredictor(ABC):
    def __init__(self, config):
        self.config = config
        self.temperature = config.get('temperature', 0.0)

    @abstractmethod
    def inference(self, example, prompt):
        pass


class MAPGDMathPredictor(MAPGDPredictor):
    def inference(self, example, prompt):
        try:
            if '{text}' in prompt:
                formatted_prompt = prompt.replace('{text}', example['text'])
            elif '{{text}}' in prompt:
                formatted_prompt = prompt.replace('{{text}}', example['text'])
            else:
                formatted_prompt = f"{prompt}\n\nProblem: {example['text']}"

            response = call_openai(
                prompt=formatted_prompt,
                system_prompt="You are a helpful assistant for solving math word problems. Show your work step by step.",
                model=self.config.get('model', "gpt-4o-mini"),
                temperature=self.temperature,
                max_tokens=self.config.get("max_tokens", 1200),
            )
            return response.strip()
        except Exception as e:
            print(f"Math prediction error: {e}")
            return "Unable to solve this problem."

    def batch_inference(self, examples, prompt, max_examples=None):
        if max_examples:
            examples = examples[:max_examples]
        return [self.inference(example, prompt) for example in examples]


class MAPGDBinaryPredictor(MAPGDPredictor):
    @staticmethod
    def _extract_structured_fields(example_text):
        fields = {'text': example_text}
        for line in example_text.splitlines():
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            fields[key.strip().lower().replace(' ', '_')] = value.strip()
        return fields

    @staticmethod
    def _fill_prompt_placeholders(prompt, fields):
        formatted_prompt = prompt
        for key, value in fields.items():
            formatted_prompt = formatted_prompt.replace(f'{{{key}}}', value)
            formatted_prompt = formatted_prompt.replace(f'{{{{{key}}}}}', value)
        if '{text}' not in prompt and '{{text}}' not in prompt:
            formatted_prompt = f"{formatted_prompt}\n\nText: {fields['text']}"
        return formatted_prompt

    @staticmethod
    def _parse_binary_label(response):
        cleaned = response.strip().upper()

        exact_patterns = [
            r"^\s*LABEL\s*:\s*['\"]?(YES|NO)['\"]?\s*$",
            r"^\s*ANSWER\s*:\s*['\"]?(YES|NO)['\"]?\s*$",
            r"^\s*['\"]?(YES|NO)['\"]?\s*$",
        ]
        for pattern in exact_patterns:
            match = re.match(pattern, cleaned)
            if match:
                return 1 if match.group(1) == 'YES' else 0

        search_patterns = [
            r"\bLABEL\s*:\s*['\"]?(YES|NO)['\"]?\b",
            r"\bANSWER\s*:\s*['\"]?(YES|NO)['\"]?\b",
            r"\b(YES|NO)\b",
        ]
        for pattern in search_patterns:
            match = re.search(pattern, cleaned)
            if match:
                return 1 if match.group(1) == 'YES' else 0

        return -1

    def inference(self, example, prompt):
        try:
            fields = self._extract_structured_fields(example['text'])
            formatted_prompt = self._fill_prompt_placeholders(prompt, fields)
            formatted_prompt += "\nOnly answer with one label: Yes or No."

            response = call_openai(
                prompt=formatted_prompt,
                system_prompt="You are a helpful assistant for text classification.",
                model=self.config.get('model', "gpt-4o-mini"),
                temperature=self.temperature,
                max_tokens=self.config.get("max_tokens", 1200),
            )
            return self._parse_binary_label(response)
        except Exception as e:
            print(f"Prediction error: {e}")
            return -1

    def batch_inference(self, examples, prompt, max_examples=None):
        if max_examples:
            examples = examples[:max_examples]
        return [self.inference(example, prompt) for example in examples]


class MAPGDAquaPredictor(MAPGDPredictor):
    def inference(self, example, prompt):
        try:
            options_text = "\n".join(example['options'])
            if '{text}' in prompt and '{options}' in prompt:
                formatted_prompt = prompt.replace('{text}', example['text'])
                formatted_prompt = formatted_prompt.replace('{options}', options_text)
            elif '{{text}}' in prompt and '{{options}}' in prompt:
                formatted_prompt = prompt.replace('{{text}}', example['text'])
                formatted_prompt = formatted_prompt.replace('{{options}}', options_text)
            else:
                formatted_prompt = f"{prompt}\n\nProblem: {example['text']}\n\nOptions:\n{options_text}"

            response = call_openai(
                prompt=formatted_prompt,
                system_prompt="You are a helpful assistant for solving math word problems with multiple choice. Show your work step by step and select the correct answer.",
                model=self.config.get('model', "gpt-4o-mini"),
                temperature=self.temperature,
                max_tokens=self.config.get("max_tokens", 1200),
            )
            return response.strip()
        except Exception as e:
            print(f"AQuA prediction error: {e}")
            return "Unable to solve this problem."

    def batch_inference(self, examples, prompt, max_examples=None):
        if max_examples:
            examples = examples[:max_examples]
        return [self.inference(example, prompt) for example in examples]


class MAPGDJailbreakPredictor(MAPGDBinaryPredictor):
    @staticmethod
    def _parse_jailbreak_label(response):
        cleaned = response.strip()
        patterns = [
            r"^\s*(?:LABEL|ANSWER)\s*:\s*['\"]?([A-Za-z]+)['\"]?\s*$",
            r"^\s*['\"]?([A-Za-z]+)['\"]?\s*$",
            r"\b(?:LABEL|ANSWER)\s*:\s*['\"]?([A-Za-z]+)['\"]?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if not match:
                continue
            label = match.group(1).lower()
            if label in {"overtattack", "suspiciouscontext", "yes"}:
                return 1
            if label in {"benign", "no"}:
                return 0
        return -1

    def inference(self, example, prompt):
        try:
            fields = self._extract_structured_fields(example["text"])
            formatted_prompt = self._fill_prompt_placeholders(prompt, fields)
            formatted_prompt += "\nReturn exactly one label allowed by the prompt. Do not add explanation."
            response = call_openai(
                prompt=formatted_prompt,
                system_prompt="You are a helpful assistant for jailbreak classification.",
                model=self.config.get("model", "gpt-4o-mini"),
                temperature=self.temperature,
                max_tokens=self.config.get("max_tokens", 1200),
            )
            return self._parse_jailbreak_label(response)
        except Exception as e:
            print(f"Jailbreak prediction error: {e}")
            return -1


def get_mapgd_predictor(task_type, config, categories=None):
    if task_type == 'binary_classification':
        return MAPGDBinaryPredictor(config)
    if task_type == 'jailbreak_classification':
        return MAPGDJailbreakPredictor(config)
    if task_type == 'math_reasoning':
        return MAPGDMathPredictor(config)
    if task_type == 'aqua_reasoning':
        return MAPGDAquaPredictor(config)
    raise ValueError(f'Unsupported task type: {task_type}')
