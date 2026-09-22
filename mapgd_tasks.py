import os
import json
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd


class MAPGDDataProcessor(ABC):
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate_prompt(self, prompt, test_examples, predictor, n=100):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass


class GSM8kTask:
    def __init__(self, data_path):
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        import json

        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_prompt(self, prompt, examples, predictor, n=50):
        correct = 0
        for ex in examples[:n]:
            question, answer = ex["question"], ex["answer"]

            prediction = predictor.predict(prompt, question)

            if str(answer).strip() in prediction:
                correct += 1

        accuracy = correct / n
        return accuracy, correct, n, prediction


class MAPGDClassificationTask(MAPGDDataProcessor):
    def evaluate_prompt(self, prompt, test_examples, predictor, n=150):
        labels = []
        preds = []
        texts = []

        test_sample = np.random.choice(test_examples, n, replace=False) if len(test_examples) > n else test_examples
        for ex in test_sample:
            try:
                pred = predictor.inference(ex, prompt)
                texts.append(ex['text'])
                labels.append(ex['label'])
                preds.append(pred)
            except Exception:
                continue

        if not labels:
            return 0.0, texts, labels, preds

        normalized_preds = [pred if pred in (0, 1) else 1 - label for label, pred in zip(labels, preds)]
        f1 = f1_score(
            labels,
            normalized_preds,
            average='binary',
            pos_label=1,
            zero_division=0,
        )
        return f1, texts, labels, preds


class MAPGDBinaryClassificationTask(MAPGDClassificationTask):
    categories = ['No', 'Yes']

    def stringify_prediction(self, pred):
        if pred not in (0, 1):
            return "Unparseable"
        return MAPGDBinaryClassificationTask.categories[pred]


class MAPGDLiarTask(MAPGDBinaryClassificationTask):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data/hf_binary")
        super().__init__(data_dir)

    def get_train_examples(self):
        try:
            exs = []
            with open(f'{self.data_dir}/train.jsonl', 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    row = json.loads(line.strip())
                    exs.append({'id': f'train-{i}', 'label': row['label'], 'text': row['text']})
            print(f"Loaded {len(exs)} training examples from liar dataset")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load Liar train data: {e}")

    def get_test_examples(self):
        try:
            exs = []
            with open(f'{self.data_dir}/test.jsonl', 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    row = json.loads(line.strip())
                    exs.append({'id': f'test-{i}', 'label': row['label'], 'text': row['text']})
            print(f"Loaded {len(exs)} test examples from liar dataset")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load Liar test data: {e}")


class MAPGDSarcasmTask(MAPGDBinaryClassificationTask):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data/sarcasm")
        super().__init__(data_dir)

    def _load_jsonl(self, split_name):
        filepath = os.path.join(self.data_dir, f"{split_name}.jsonl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Missing Sarcasm split: {filepath}. Expected JSONL rows with text and binary label."
            )
        examples = []
        with open(filepath, "r", encoding="utf-8") as source:
            for index, line in enumerate(source):
                if not line.strip():
                    continue
                row = json.loads(line)
                examples.append(
                    {
                        "id": row.get("id", f"{split_name}-{index}"),
                        "text": row["text"],
                        "label": int(row["label"]),
                    }
                )
        return examples

    def get_train_examples(self):
        return self._load_jsonl("train")

    def get_test_examples(self):
        return self._load_jsonl("test")


class MAPGDJailbreakTask(MAPGDBinaryClassificationTask):
    def __init__(self, data_dir=None):
        data_dir = os.path.join(os.path.dirname(__file__), "data/jailbreak")
        super().__init__(data_dir)

    import json
    import os

    def _load_tsv(self, filepath, split_name):
        exs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    print(f"Skipping line {i}: empty line")
                    continue
                try:
                    if "\t" in line:
                        text_json, label = line.rsplit("\t", 1)
                    elif "\\t" in line:
                        text_json, label = line.rsplit("\\t", 1)
                    else:
                        print(f"Skipping line {i}: no tab separator found: {repr(line)}")
                        continue
                    try:
                        payload = json.loads(text_json)
                    except json.JSONDecodeError:
                        if text_json.endswith("]") and not text_json.startswith("["):
                            payload = json.loads(text_json[:-1])
                        else:
                            raise
                    text_list = payload if isinstance(payload, list) else [payload]
                    if not text_list or not isinstance(text_list[0], dict):
                        print(f"Skipping line {i}: invalid JSON message payload: {text_json}")
                        continue
                    if "text" not in text_list[0]:
                        print(f"Skipping line {i}: missing 'text' field in JSON: {text_json}")
                        continue
                    text = text_list[0]["text"]
                    exs.append({'id': f'{split_name}-{i}', 'label': int(label), 'text': text})
                except Exception as e:
                    print(f"Skipping line {i} due to error: {e}, line: {repr(line)}")
                    continue
        print(f"Loaded {len(exs)} {split_name} examples from jailbreak dataset")
        return exs

    def get_train_examples(self):
        return self._load_tsv(os.path.join(self.data_dir, "train.tsv"), "train")

    def get_test_examples(self):
        return self._load_tsv(os.path.join(self.data_dir, "test.tsv"), "test")


class MAPGDEthosBinaryTask(MAPGDBinaryClassificationTask):
    data_dir = os.path.join(os.path.dirname(__file__), "data/ethos")
    categories = ['No', 'Yes']

    def _load_data(self):
        df = pd.read_csv(
            os.path.join(self.data_dir, 'Ethos_Dataset_Binary.csv'), sep=';', header=None, names=['text', 'score']
        )

        df = df[(df['score'] <= 0) | (df['score'] >= 0.7)].reset_index(drop=True)
        return df

    def get_train_examples(self):
        df = self._load_data()
        exs = df.reset_index().to_dict('records')

        exs = [{'id': x['index'], 'text': x['text'], 'label': 1 if x['score'] > 0.4 else 0} for x in exs[200:]]
        print(f"Loaded {len(exs)} training examples")
        return exs

    def get_test_examples(self):
        df = self._load_data()
        exs = df.reset_index().to_dict('records')

        exs = [{'id': x['index'], 'text': x['text'], 'label': 1 if x['score'] > 0.4 else 0} for x in exs[:200]]
        print(f"Loaded {len(exs)} test examples")
        return exs


import re


class MAPGDGsm8kTask(MAPGDDataProcessor):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data/gsm8k")
        super().__init__(data_dir)
        self.answer_pattern = re.compile(r'#### (-?\d+(?:\.\d+)?)')

    def _load_parquet_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load {filename}: {e}")

    def _extract_numerical_answer(self, answer_text):
        match = self.answer_pattern.search(answer_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _extract_prediction_answer(self, prediction_text):
        match = self.answer_pattern.search(prediction_text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass

        prediction_text = prediction_text.replace(',', '')

        numbers = re.findall(r'-?\d+\.\d+|-?\d+', prediction_text)

        if numbers:
            try:
                return float(numbers[-1])
            except (ValueError, IndexError):
                return None

        return None

    def get_train_examples(self):
        try:
            df = self._load_parquet_data("train.parquet")
            exs = []
            for i, row in df.iterrows():
                answer_num = self._extract_numerical_answer(row['answer'])
                if answer_num is not None:
                    exs.append(
                        {
                            'id': f'train-{i}',
                            'text': row['question'],
                            'answer': row['answer'],
                            'numerical_answer': answer_num,
                        }
                    )
            print(f"Loaded {len(exs)} training examples from GSM8k dataset")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load GSM8k train data: {e}")

    def get_test_examples(self):
        try:
            df = self._load_parquet_data("test.parquet")
            exs = []
            for i, row in df.iterrows():
                answer_num = self._extract_numerical_answer(row['answer'])
                if answer_num is not None:
                    exs.append(
                        {
                            'id': f'test-{i}',
                            'text': row['question'],
                            'answer': row['answer'],
                            'numerical_answer': answer_num,
                        }
                    )
            print(f"Loaded {len(exs)} test examples from GSM8k dataset")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load GSM8k test data: {e}")

    def evaluate_prompt(self, prompt, test_examples, predictor, n=50):
        correct = 0
        total = 0
        texts = []
        true_answers = []
        pred_answers = []

        test_sample = np.random.choice(test_examples, n, replace=False) if len(test_examples) > n else test_examples

        for ex in test_sample:
            try:
                prediction_text = predictor.inference(ex, prompt)

                pred_answer = self._extract_prediction_answer(prediction_text)
                true_answer = ex['numerical_answer']

                texts.append(ex['text'])
                true_answers.append(true_answer)
                pred_answers.append(pred_answer)

                if pred_answer is not None and abs(pred_answer - true_answer) < 1e-6:
                    correct += 1

                total += 1

            except Exception as e:
                print(f"Error processing example {ex['id']}: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0
        print(f"GSM8k Accuracy: {correct}/{total} = {accuracy:.4f}")

        return accuracy, texts, true_answers, pred_answers

    def stringify_prediction(self, pred):
        if pred is None:
            return "No answer"
        return str(pred)


class MAPGDAquaTask(MAPGDDataProcessor):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data/aqua")
        super().__init__(data_dir)
        self.answer_pattern = re.compile(r'Answer\s*:\s*([A-E])', re.IGNORECASE)

    def _load_parquet_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        try:
            df = pd.read_parquet(filepath, engine='pyarrow')

            data = df.to_dict('records')
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load {filepath}: {e}")

    def _format_options(self, options):
        return "\n".join(options)

    def _extract_answer_from_prediction(self, prediction_text):
        match = self.answer_pattern.search(prediction_text)
        if match:
            return match.group(1).upper()

        letters = re.findall(r'\b([A-E])\b', prediction_text.upper())
        if letters:
            return letters[-1]

        return None

    def get_train_examples(self):
        try:
            data = self._load_parquet_data("train.parquet")
            exs = []
            for i, item in enumerate(data):
                exs.append(
                    {
                        'id': f'train-{i}',
                        'text': item['question'],
                        'options': item['options'],
                        'rationale': item.get('rationale', ''),
                        'correct': item['correct'],
                    }
                )
            print(f"Loaded {len(exs)} training examples from AQuA dataset")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load AQuA train data: {e}")

    def get_test_examples(self):
        try:
            data = self._load_parquet_data("test.parquet")
            exs = []
            for i, item in enumerate(data):
                exs.append(
                    {
                        'id': f'test-{i}',
                        'text': item['question'],
                        'options': item['options'],
                        'rationale': item.get('rationale', ''),
                        'correct': item['correct'],
                    }
                )
            print(f"Loaded {len(exs)} test examples from AQuA dataset")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load AQuA test data: {e}")

    def evaluate_prompt(self, prompt, test_examples, predictor, n=50):
        correct = 0
        total = 0
        texts = []
        true_answers = []
        pred_answers = []

        test_sample = (
            np.random.choice(test_examples, n, replace=False).tolist() if len(test_examples) > n else test_examples
        )

        for ex in test_sample:
            try:
                prediction_text = predictor.inference(ex, prompt)
                pred_answer = self._extract_answer_from_prediction(prediction_text)
                true_answer = ex['correct']

                texts.append(ex['text'])
                true_answers.append(true_answer)
                pred_answers.append(pred_answer)

                if pred_answer is not None and pred_answer == true_answer:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"Error processing example {ex.get('id', 'N/A')}: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0
        print(f"AQuA Accuracy: {correct}/{total} = {accuracy:.4f}")

        return accuracy, texts, true_answers, pred_answers

    def stringify_prediction(self, pred):
        if pred is None:
            return "No answer"
        return str(pred)


class MAPGDSvampTask(MAPGDDataProcessor):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data/svamp")
        super().__init__(data_dir)
        self.answer_pattern = re.compile(r'#### (-?\d+(?:\.\d+)?)')

    def _load_json_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load {filename}: {e}")

    def _extract_prediction_answer(self, prediction_text):
        match = self.answer_pattern.search(prediction_text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass

        prediction_text = prediction_text.replace(',', '')
        numbers = re.findall(r'-?\d+\.\d+|-?\d+', prediction_text)

        if numbers:
            try:
                return float(numbers[-1])
            except (ValueError, IndexError):
                return None

        return None

    def get_train_examples(self):
        try:
            data = self._load_json_data("train.json")
            exs = []
            for i, item in enumerate(data):
                exs.append(
                    {
                        'id': item.get('ID', f'train-{i}'),
                        'text': item['Body'],
                        'question': item['Question'],
                        'answer': item['Answer'],
                        'numerical_answer': float(item['Answer']),
                        'type': item.get('Type', 'Unknown'),
                    }
                )
            print(f"Loaded {len(exs)} training examples from SVAMP dataset")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load SVAMP train data: {e}")

    def get_test_examples(self):
        try:
            data = self._load_json_data("test.json")
            exs = []
            for i, item in enumerate(data):
                exs.append(
                    {
                        'id': item.get('ID', f'test-{i}'),
                        'text': item['Body'],
                        'question': item['Question'],
                        'answer': item['Answer'],
                        'numerical_answer': float(item['Answer']),
                        'type': item.get('Type', 'Unknown'),
                    }
                )
            print(f"Loaded {len(exs)} test examples from SVAMP dataset")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load SVAMP test data: {e}")

    def evaluate_prompt(self, prompt, test_examples, predictor, n=50):
        correct = 0
        total = 0
        texts = []
        true_answers = []
        pred_answers = []

        test_sample = np.random.choice(test_examples, n, replace=False) if len(test_examples) > n else test_examples

        for ex in test_sample:
            try:
                ex_with_question = ex.copy()
                ex_with_question['text'] = f"{ex['text']} {ex['question']}"

                prediction_text = predictor.inference(ex_with_question, prompt)

                pred_answer = self._extract_prediction_answer(prediction_text)
                true_answer = ex['numerical_answer']

                texts.append(f"{ex['text']} {ex['question']}")
                true_answers.append(true_answer)
                pred_answers.append(pred_answer)

                if pred_answer is not None and abs(pred_answer - true_answer) < 1e-6:
                    correct += 1

                total += 1

            except Exception as e:
                print(f"Error processing example {ex['id']}: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0
        print(f"SVAMP Accuracy: {correct}/{total} = {accuracy:.4f}")

        return accuracy, texts, true_answers, pred_answers

    def stringify_prediction(self, pred):
        if pred is None:
            return "No answer"
        return str(pred)


def get_mapgd_task_class(task_name):
    if task_name == 'liar':
        return MAPGDLiarTask
    elif task_name == 'jailbreak':
        return MAPGDJailbreakTask
    elif task_name == 'ethos':
        return MAPGDEthosBinaryTask
    elif task_name == 'gsm8k':
        return MAPGDGsm8kTask
    elif task_name == 'aqua':
        return MAPGDAquaTask
    elif task_name == 'svamp':
        return MAPGDSvampTask
    elif task_name == 'sarcasm':
        return MAPGDSarcasmTask
    else:
        raise ValueError(f'Unsupported task: {task_name}')
