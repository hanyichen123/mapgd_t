import argparse
import re
from importlib import import_module

import numpy as np
from sentence_transformers import SentenceTransformer


def load_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--task',
        type=str,
        default='liar',
        choices=['liar', 'ethos', 'jailbreak', 'sarcasm', 'gsm8k', 'aqua', 'svamp'],
        help="Configuration source to use (liar, ethos, jailbreak, gsm8k, aqua or svamp)",
    )
    args, _ = parser.parse_known_args()
    config_source = args.task or 'liar'

    config = import_module(f'{config_source}_config')
    print(f"Using {config_source}_config for DATA_CONFIG and EXPERIMENT_CONFIG")
    return config.DATA_CONFIG, config.EXPERIMENT_CONFIG


try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    GPU_AVAILABLE = False


class SemanticModelManager:
    _instance = None
    _semantic_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        if self._semantic_model is None:
            self._semantic_model = self._load_semantic_model()
        return self._semantic_model

    def _load_semantic_model(self):
        try:
            DATA_CONFIG, _ = load_config()
            if torch is not None:
                device = 'cuda' if GPU_AVAILABLE else 'cpu'
                model = SentenceTransformer(DATA_CONFIG['sentence_transformer_model'], device=device)
                print(f"SemanticModelManager: Loaded SentenceTransformer on {device}")
                return model
        except Exception as e:
            print(f"Warning: Failed to load SentenceTransformer: {e}")
            return None


class TextUtils:
    @staticmethod
    def extract_gradient_text(gradient):
        if isinstance(gradient, str):
            return gradient
        elif isinstance(gradient, list):
            return " ".join(gradient)
        elif isinstance(gradient, dict):
            return gradient.get('content', str(gradient))
        else:
            return str(gradient)

    @staticmethod
    def parse_llm_response_with_tags(response, tag_pattern):
        matches = re.findall(tag_pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return [match.strip() for match in matches]
        else:
            return [response.strip()]

    @staticmethod
    def format_conflict_info(conflicts):
        if not conflicts:
            return "无冲突"

        conflict_texts = []
        for conflict in conflicts:
            sim_score = conflict.get('sim', conflict.get('similarity', 0))
            conflict_texts.append(
                f"智能体 {conflict['agents'][0]} 和 {conflict['agents'][1]} 的建议存在冲突 (相似度: {sim_score:.2f})"
            )

        return "\n".join(conflict_texts)


class VectorUtils:
    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def encode_texts(texts, fallback_dim=384):
        model_manager = SemanticModelManager()
        model = model_manager.get_model()

        if model is not None:
            return model.encode(texts, convert_to_numpy=True)
        else:
            print(f"Warning: Using random vectors as fallback for {len(texts)} texts")
            return np.random.rand(len(texts), fallback_dim)

    @staticmethod
    def batch_cosine_similarity_matrix(vectors_a, vectors_b):
        norm_a = np.linalg.norm(vectors_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(vectors_b, axis=1, keepdims=True)

        norm_a = np.where(norm_a == 0, 1, norm_a)
        norm_b = np.where(norm_b == 0, 1, norm_b)

        normalized_a = vectors_a / norm_a
        normalized_b = vectors_b / norm_b

        return normalized_a @ normalized_b.T


class ParsingUtils:
    @staticmethod
    def parse_gradient_response(response):
        pattern = r'<\s*START\s*>(.*?)<\s*END\s*>'
        results = TextUtils.parse_llm_response_with_tags(response, pattern)
        return results if results else [response.strip()]

    @staticmethod
    def parse_fusion_response(response):
        pattern = r'<\s*START\s*>(.*?)<\s*END\s*>'
        results = TextUtils.parse_llm_response_with_tags(response, pattern)
        return results[0]

    @staticmethod
    def get_parsing_rules():
        pattern = r'<\s*START\s*>(.*?)<\s*END\s*>'
        return {
            'gradient_blocks': {
                'pattern': pattern,
                'flags': ['DOTALL', 'IGNORECASE'],
                'postprocess': 'strip each extracted block',
                'fallback': 'return the stripped raw response as a single gradient if no tagged block is found',
            },
            'fusion_blocks': {
                'pattern': pattern,
                'flags': ['DOTALL', 'IGNORECASE'],
                'postprocess': 'strip extracted blocks and keep the first fused block',
                'fallback': 'use the first parsed block returned by the generic tag parser',
            },
            'format_specification': {
                'gradient_generation': 'Wrap each reason with <START> and <END>',
                'gradient_fusion': 'Wrap the unified improvement with <START> and <END>',
                'prompt_application': 'Wrap each improved prompt with <START> and <END>',
            },
        }


class DiversityFilter:
    @staticmethod
    def filter_by_semantic_similarity(candidates, diversity_threshold=0.7, max_candidates=None):
        if not candidates or len(candidates) == 1:
            return candidates

        diverse_candidates = [candidates[0]]

        model_manager = SemanticModelManager()
        model = model_manager.get_model()

        if model is not None:
            print("Using semantic similarity for diversity filtering...")

            candidate_embeddings = VectorUtils.encode_texts(candidates)

            for i, candidate in enumerate(candidates[1:], 1):
                candidate_embedding = candidate_embeddings[i]

                is_diverse = True
                for existing_candidate in diverse_candidates:
                    existing_idx = candidates.index(existing_candidate)
                    existing_embedding = candidate_embeddings[existing_idx]

                    similarity = VectorUtils.cosine_similarity(candidate_embedding, existing_embedding)

                    if similarity > diversity_threshold:
                        is_diverse = False
                        break

                if is_diverse:
                    diverse_candidates.append(candidate)

                if max_candidates and len(diverse_candidates) >= max_candidates:
                    break

            print(f"Semantic diversity filtering: {len(diverse_candidates)}/{len(candidates)} candidates retained")
        else:
            print("No semantic model, using simple diversity filtering...")

            for candidate in candidates[1:]:
                if max_candidates and len(diverse_candidates) >= max_candidates:
                    break

                if candidate not in diverse_candidates:
                    diverse_candidates.append(candidate)

        return diverse_candidates


class MAPGDUtils:
    @staticmethod
    def get_semantic_model():
        return SemanticModelManager().get_model()

    @staticmethod
    def extract_gradient_text(gradient):
        return TextUtils.extract_gradient_text(gradient)

    @staticmethod
    def parse_llm_response_with_tags(response, tag_pattern):
        return TextUtils.parse_llm_response_with_tags(response, tag_pattern)

    @staticmethod
    def format_conflicts(conflicts):
        return TextUtils.format_conflict_info(conflicts)

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        return VectorUtils.cosine_similarity(vec_a, vec_b)

    @staticmethod
    def encode_texts(texts):
        return VectorUtils.encode_texts(texts)

    @staticmethod
    def parse_gradient_response(response):
        return ParsingUtils.parse_gradient_response(response)

    @staticmethod
    def parse_fusion_response(response):
        return ParsingUtils.parse_fusion_response(response)

    @staticmethod
    def get_parsing_rules():
        return ParsingUtils.get_parsing_rules()

    @staticmethod
    def filter_for_diversity(candidates, diversity_threshold=0.7, max_candidates=None):
        return DiversityFilter.filter_by_semantic_similarity(candidates, diversity_threshold, max_candidates)
