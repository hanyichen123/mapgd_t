from utils import load_config

DATA_CONFIG, EXPERIMENT_CONFIG = load_config()

import time
import random
import numpy as np
import re
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from llm import call_openai
from utils import MAPGDUtils
from abc import ABCMeta, abstractmethod
from hcgc_caaw import HypersphereConstrainedGradientClustering

try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    GPU_AVAILABLE = False


class MAPGDFramework(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config
        self.max_iterations = config.get('max_iterations', 10)

        self.agent_manager = AgentManager(config)

        self.gradient_coordinator = GradientCoordinator(config)

        self.prompt_expander = CollaborativePromptExpander(config)

        self.candidate_selector = BanditBasedSelector(config)

        self.convergence_monitor = ConvergenceMonitor(config)

        print(f"MAPGD Framework initialized with {config.get('num_agents', 4)} specialized agents")

    @abstractmethod
    def optimize_prompt(self, initial_prompt, task_name, data_dir, predictor=None):
        pass


class SpecializedPromptAgent:
    BASE_AGENT_TYPES = {
        'instruction_specialist': {
            'focus': 'Clarity of task descriptions and instructions',
            'expertise': 'Analyze the completeness, clarity, and enforceability of instructions',
            'gradient_type': 'instruction_gradient',
            'role_description': 'Specializes in analyzing and improving task instructions, ensuring clarity, completeness, and executability',
        },
        'example_curator': {
            'focus': 'Few-shot example selection and formatting',
            'expertise': 'Optimize the representativeness, diversity, and format consistency of examples',
            'gradient_type': 'example_gradient',
            'role_description': 'Focuses on selecting representative, diverse examples and ensuring consistent formatting',
        },
        'format_designer': {
            'focus': 'Output format and structure design',
            'expertise': 'Design clear output templates and structured formats',
            'gradient_type': 'format_gradient',
            'role_description': 'Designs clear output templates and structured formats for better model understanding',
        },
        'style_optimizer': {
            'focus': 'Language style and tone adjustments',
            'expertise': 'Optimize the professionalism and adaptability of language expression',
            'gradient_type': 'style_gradient',
            'role_description': 'Optimizes language expression for professionalism and task-specific adaptation',
        },
        'generic': {
            'focus': 'General prompt improvement',
            'expertise': 'Provide general reasons and improvements without specialization',
            'gradient_type': 'generic_gradient',
            'role_description': 'General agent without specialization, explores improvements broadly',
        },
    }

    MATH_AGENT_TYPES = {
        'reasoning_specialist': {
            'focus': 'Mathematical reasoning process and step-by-step logic',
            'expertise': 'Analyze and improve the logical flow, step clarity, and reasoning structure in mathematical problem solving',
            'gradient_type': 'reasoning_gradient',
            'role_description': 'Specializes in enhancing mathematical reasoning processes, ensuring clear step-by-step logic and proper problem decomposition',
        },
        'calculation_optimizer': {
            'focus': 'Numerical computation accuracy and methodology',
            'expertise': 'Optimize calculation methods, numerical precision, and computational approaches',
            'gradient_type': 'calculation_gradient',
            'role_description': 'Focuses on improving calculation accuracy, suggesting better computational methods and ensuring numerical correctness',
        },
        'problem_interpreter': {
            'focus': 'Word problem comprehension and key information extraction',
            'expertise': 'Enhance problem understanding, identify key variables, constraints, and mathematical relationships',
            'gradient_type': 'interpretation_gradient',
            'role_description': 'Specializes in interpreting math word problems, extracting relevant information, and identifying mathematical relationships',
        },
        'solution_formatter': {
            'focus': 'Mathematical solution presentation and answer format',
            'expertise': 'Design clear solution formats, ensure proper mathematical notation, and structure final answers',
            'gradient_type': 'format_gradient',
            'role_description': 'Optimizes mathematical solution presentation, ensures clear formatting and proper answer notation (e.g., #### format)',
        },
        'generic': {
            'focus': 'General mathematical prompt improvement',
            'expertise': 'Provide general mathematical reasoning improvements without specialization',
            'gradient_type': 'generic_gradient',
            'role_description': 'General mathematical agent without specialization, explores broad mathematical reasoning improvements',
        },
    }

    TASK_AGENT_MAPPING = {
        'liar': BASE_AGENT_TYPES,
        'ethos': BASE_AGENT_TYPES,
        'jailbreak': BASE_AGENT_TYPES,
        'gsm8k': MATH_AGENT_TYPES,
        'aqua': MATH_AGENT_TYPES,
        'svamp': MATH_AGENT_TYPES,
        'classification': BASE_AGENT_TYPES,
        'default': BASE_AGENT_TYPES,
    }
    GRADIENT_SYSTEM_PROMPT = "You are a professional prompt word optimization expert."
    GRADIENT_USER_TEMPLATE = """
I'm trying to write a zero-shot {task_context} prompt.
My current prompt is: "{prompt}"
But this prompt gets the following examples wrong:
{error_text}

As an expert specialized in {expertise}, please give {num_feedbacks} reasons
why the prompt could have gotten these examples wrong from a {focus} perspective.
Focus specifically on {task_context} requirements.

Wrap each reason with <START> and <END>
""".strip()

    def __init__(self, agent_type, task_name='default'):
        self.task_name = task_name
        self.agent_type = agent_type

        agent_types = self.TASK_AGENT_MAPPING.get(task_name, self.BASE_AGENT_TYPES)

        if agent_type in agent_types:
            self.config = agent_types[agent_type]
        else:
            print(f"Warning: Unknown agent type '{agent_type}' for task '{task_name}', using generic")
            self.config = agent_types.get('generic', self.BASE_AGENT_TYPES['generic'])

        self.current_prompt = None
        self.gradient_history = []
        self.performance_memory = []
        self.specialization_confidence = 1.0

        print(f"Initialized {agent_type} for {task_name}: {self.config['role_description']}")

    @classmethod
    def get_available_agent_types(cls, task_name='default'):
        agent_types = cls.TASK_AGENT_MAPPING.get(task_name, cls.BASE_AGENT_TYPES)
        return list(agent_types.keys())

    def _get_task_context(self):
        if self.task_name == 'gsm8k' or 'math' in self.task_name:
            return "mathematical word problem solving"
        return "text classification"

    def get_gradient_system_prompt(self):
        return self.GRADIENT_SYSTEM_PROMPT

    def get_gradient_prompt_template(self):
        return self.GRADIENT_USER_TEMPLATE

    def get_reproducibility_manifest(self):
        return {
            'agent_type': self.agent_type,
            'task_name': self.task_name,
            'focus': self.config['focus'],
            'expertise': self.config['expertise'],
            'role_description': self.config['role_description'],
            'gradient_system_prompt': self.get_gradient_system_prompt(),
            'gradient_user_template': self.get_gradient_prompt_template(),
            'gradient_user_template_placeholders': [
                'task_context',
                'prompt',
                'error_text',
                'expertise',
                'focus',
                'num_feedbacks',
            ],
            'gradient_output_format': "Wrap each reason with <START> and <END>",
            'resolved_task_context': self._get_task_context(),
        }

    async def generate_specialized_gradient_async(self, prompt, error_examples, task):
        gradient_prompt = self._construct_gradient_prompt(prompt, error_examples, task)

        loop = asyncio.get_event_loop()

        def sync_llm_call():
            return call_openai(prompt=gradient_prompt, system_prompt=self.get_gradient_system_prompt())

        with ThreadPoolExecutor(max_workers=EXPERIMENT_CONFIG['max_concurrent_agents']) as executor:
            response = await loop.run_in_executor(executor, sync_llm_call)

        gradient = self._parse_gradient_response(response)

        self.gradient_history.append(
            {
                'prompt': prompt,
                'gradient': gradient,
                'context': error_examples,
                'timestamp': time.time(),
                'agent_type': self.agent_type,
            }
        )

        return gradient

    def generate_specialized_gradient(self, prompt, error_examples, task):
        gradient_prompt = self._construct_gradient_prompt(prompt, error_examples, task)

        response = call_openai(prompt=gradient_prompt, system_prompt=self.get_gradient_system_prompt())

        gradient = self._parse_gradient_response(response)

        self.gradient_history.append(
            {'prompt': prompt, 'gradient': gradient, 'context': error_examples, 'timestamp': time.time()}
        )

        return gradient

    def _construct_gradient_prompt(self, prompt, error_examples, task, num_feedbacks=4):
        error_text = self._format_error_examples(error_examples)

        task_context = self._get_task_context()
        return self.get_gradient_prompt_template().format(
            task_context=task_context,
            prompt=prompt,
            error_text=error_text,
            expertise=self.config['expertise'],
            focus=self.config['focus'],
            num_feedbacks=num_feedbacks,
        )

    def _format_error_examples(self, error_examples):
        texts = []
        if not error_examples:
            return "暂无错误样本"

        for example in error_examples:
            text = example.get('text', '')
            texts.append(text)
        texts = (
            np.random.choice(texts, EXPERIMENT_CONFIG['error_group_size'], replace=False)
            if len(texts) > EXPERIMENT_CONFIG['error_group_size']
            else texts
        )
        return "\n".join(texts)

    def _parse_gradient_response(self, response):
        return MAPGDUtils.parse_gradient_response(response)


class GradientCoordinator:
    def __init__(self, config):
        self.fusion_method = config.get('fusion_method', 'semantic_clustering')
        self.conflict_threshold = config.get('conflict_threshold', 0.3)
        self.max_clusters = config.get('max_clusters', 5)
        self.log_gradients = config.get('log_gradients', False)
        self.last_coordination_log = None

        self.semantic_model = MAPGDUtils.get_semantic_model()
        self.hcgc = HypersphereConstrainedGradientClustering(config)
        self.enable_hcgc = config.get('enable_hcgc', True)
        self.enable_caaw = config.get('enable_caaw', True)

    def get_reproducibility_manifest(self):
        return {
            'fusion_method': self.fusion_method,
            'conflict_threshold': self.conflict_threshold,
            'max_clusters': self.max_clusters,
            'directional_conflict_check': getattr(self.hcgc, 'enable_directional_conflict_check', False),
            'fusion_output_format': "Wrap each unified and coherent prompt improvement with <START> and <END>",
        }

    def coordinate_gradients(self, agent_gradients, caaw_instance=None, agent_history=None, current_iteration=0):
        print(f"Coordinating gradients from {len(agent_gradients)} agents...")
        self.last_coordination_log = None
        raw_gradient_snapshot = {
            agent_id: [MAPGDUtils.extract_gradient_text(g) for g in grads]
            for agent_id, grads in agent_gradients.items()
        }

        print("[HCGC] Using hypersphere-constrained clustering...")

        gradient_vectors, metadata = self.hcgc._embed_to_hypersphere(agent_gradients)
        if len(gradient_vectors) == 0:
            if self.log_gradients:
                self.last_coordination_log = {
                    'raw_agent_gradients': raw_gradient_snapshot,
                    'clusters': [],
                    'fused_gradients': [],
                    'conflicts': [],
                    'pair_analysis': [],
                    'complementary_pairs': [],
                    'caaw_records': getattr(caaw_instance, "last_fusion_records", []) if caaw_instance else [],
                    'iteration': current_iteration,
                }
            return []

        conflicts = self.hcgc._detect_conflicts(gradient_vectors, metadata)
        print(f"[HCGC] Detected {len(conflicts)} gradient conflicts (threshold={self.hcgc.conflict_threshold})")
        if hasattr(self.hcgc, "_log_conflicts"):
            self.hcgc._log_conflicts(conflicts, metadata)
        n_clusters = min(len(gradient_vectors), self.max_clusters)
        cluster_labels, centroids = self.hcgc._cluster_gradients(gradient_vectors, n_clusters)
        cluster_labels = self.hcgc._apply_angular_margin(gradient_vectors, cluster_labels, centroids)
        clusters = self.hcgc._organize_clusters(metadata, cluster_labels, centroids)
        print(f"[HCGC] Formed {len(clusters)} coherent gradient clusters.")

        if self.enable_caaw and caaw_instance and agent_history is not None:
            print("[CAAW] Applying Channel-Adaptive Agent Weighting...")
            fused_gradients = self._fuse_clusters_with_caaw(
                clusters, conflicts, agent_gradients, caaw_instance, agent_history, current_iteration
            )
        else:
            print("[Fusion] Using uniform (non-weighted) fusion.")
            fused_gradients = self.hcgc._fuse_clusters(clusters, conflicts)

        print(f"Fusion complete: generated {len(fused_gradients)} coordinated gradients.")

        if self.log_gradients:
            cluster_snapshot = []
            for cluster in clusters:
                cluster_snapshot.append(
                    {
                        'id': cluster.get('id'),
                        'agents': cluster.get('agents', []),
                        'gradients': [MAPGDUtils.extract_gradient_text(g) for g in cluster.get('gradients', [])],
                    }
                )
            fused_snapshot = [MAPGDUtils.extract_gradient_text(g) for g in fused_gradients]
            self.last_coordination_log = {
                'raw_agent_gradients': raw_gradient_snapshot,
                'clusters': cluster_snapshot,
                'fused_gradients': fused_snapshot,
                'conflicts': conflicts,
                'pair_analysis': getattr(self.hcgc, "last_pair_analysis", []),
                'complementary_pairs': getattr(self.hcgc, "last_complementary_pairs", []),
                'caaw_records': getattr(caaw_instance, "last_fusion_records", []) if caaw_instance else [],
                'iteration': current_iteration,
            }
        return fused_gradients

    def _fuse_clusters_with_caaw(self, clusters, conflicts, original_agent_gradients, caaw, history, iteration):
        fused_gradients = []
        caaw.last_fusion_records = []
        caaw.last_fusion_agents = set()

        agent_to_gradient_map = {}
        for agent_id, grads in original_agent_gradients.items():
            agent_to_gradient_map[agent_id] = [MAPGDUtils.extract_gradient_text(g) for g in grads]

        for cluster in clusters:
            agent_ids_in_cluster = list(cluster['agents'])
            agent_weights = (
                caaw.compute_agent_weights(agent_ids_in_cluster, history, iteration) if agent_ids_in_cluster else {}
            )
            if not agent_weights:
                default_weight = 1.0 / max(1, len(cluster['gradients']))
                if agent_ids_in_cluster:
                    agent_weights = {aid: default_weight for aid in agent_ids_in_cluster}
                else:
                    agent_weights = {'unknown': default_weight}

            if len(cluster['gradients']) == 1:
                fused_gradient = cluster['gradients'][0]
            else:
                fused_gradient = caaw.weighted_fusion(
                    gradients=cluster['gradients'],
                    agent_weights=agent_weights,
                    agent_to_gradient_map=agent_to_gradient_map,
                )

            fused_gradients.append(fused_gradient)
            caaw.last_fusion_agents.update(agent_ids_in_cluster)
            caaw.last_fusion_records.append(
                {
                    'iteration': iteration,
                    'cluster_id': cluster.get('id'),
                    'agents': agent_ids_in_cluster,
                    'agent_weights': agent_weights,
                    'weighting_mode': getattr(caaw, 'last_weighting_mode', 'uniform'),
                    'weighting_reason': getattr(caaw, 'last_weighting_reason', 'unknown'),
                    'cluster_size': len(cluster['gradients']),
                    'fused_gradient': MAPGDUtils.extract_gradient_text(fused_gradient),
                }
            )

        return fused_gradients

    def _vectorize_gradients(self, agent_gradients):
        vectors = {}

        all_texts = []
        text_to_agent = {}

        for agent_id, gradients in agent_gradients.items():
            for i, gradient in enumerate(gradients):
                gradient_text = MAPGDUtils.extract_gradient_text(gradient)
                all_texts.append(gradient_text)
                text_to_agent[len(all_texts) - 1] = (agent_id, i)

        print(f"Batch encoding {len(all_texts)} gradient texts...")
        all_vectors = MAPGDUtils.encode_texts(all_texts)

        for agent_id in agent_gradients.keys():
            vectors[agent_id] = []

        for idx, vector in enumerate(all_vectors):
            agent_id, _ = text_to_agent[idx]
            vectors[agent_id].append(vector)

        for agent_id in vectors:
            vectors[agent_id] = np.array(vectors[agent_id])

        return vectors

    def _detect_gradient_conflicts(self, gradient_vectors):
        conflicts = []

        agent_ids = list(gradient_vectors.keys())

        for i, agent_a in enumerate(agent_ids):
            for agent_b in agent_ids[i + 1 :]:
                vectors_a = gradient_vectors[agent_a]
                vectors_b = gradient_vectors[agent_b]
                sim_mat = (
                    vectors_a
                    @ vectors_b.T
                    / (np.linalg.norm(vectors_a, axis=1)[:, None] * np.linalg.norm(vectors_b, axis=1)[None, :])
                )
                idx = np.where(sim_mat < -self.conflict_threshold)
                for i_a, i_b in zip(*idx):
                    conflicts.append({'agents': (agent_a, agent_b), 'indices': (i_a, i_b), 'sim': sim_mat[i_a, i_b]})
        return conflicts

    def _cluster_gradients(self, agent_gradients, gradient_vectors):
        all_vectors = []
        gradient_metadata = []

        for agent_id, gradients in agent_gradients.items():
            vectors = gradient_vectors[agent_id]
            for i, (gradient, vector) in enumerate(zip(gradients, vectors)):
                all_vectors.append(vector)
                gradient_metadata.append({'agent_id': agent_id, 'gradient_idx': i, 'gradient': gradient})

        n_clusters = min(len(all_vectors), 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(all_vectors)

        clusters = []
        for cluster_id in range(n_clusters):
            cluster_gradients = []
            cluster_agents = set()

            for i, label in enumerate(cluster_labels):
                if label == cluster_id:
                    metadata = gradient_metadata[i]
                    cluster_gradients.append(metadata['gradient'])
                    cluster_agents.add(metadata['agent_id'])

            clusters.append(
                {
                    'id': cluster_id,
                    'gradients': cluster_gradients,
                    'agents': list(cluster_agents),
                    'centroid': kmeans.cluster_centers_[cluster_id],
                }
            )

        return clusters

    def _fuse_gradient_clusters(self, clusters, conflicts):
        fused_gradients = []

        for cluster in clusters:
            if len(cluster['gradients']) == 1:
                fused_gradients.append(cluster['gradients'][0])
            else:
                fusion_prompt = self._create_fusion_prompt(cluster['gradients'], conflicts)

                response = call_openai(
                    prompt=fusion_prompt,
                    system_prompt="You are a professional prompt word optimization expert who is good at integrating multiple improvement suggestions.",
                )

                fused_gradient = self._parse_fusion_response(response)
                fused_gradients.append(fused_gradient)

        return fused_gradients

    def _create_fusion_prompt(self, gradients, conflicts, num_gradients=4):
        gradient_texts = []
        for i, gradient in enumerate(gradients):
            gradient_text = MAPGDUtils.extract_gradient_text(gradient)
            gradient_texts.append(f"建议{i + 1}: {gradient_text}")

        conflict_info = ""
        if conflicts:
            conflict_info = f"""

            The following potential conflicts have been detected and need to be resolved:
            {self._format_conflicts(conflicts)}
            """
        gradients = "\n".join(gradient_texts)
        fusion_prompt = f"""

        I need to combine the following multiple prompt improvement suggestions into {num_gradients} unified 
        and coherent prompt improvements:

        {gradients}
        {conflict_info}

        Wrap each unified and coherent prompt improvement with <START> and <END>
        """

        return fusion_prompt

    def _parse_fusion_response(self, response):
        return MAPGDUtils.parse_fusion_response(response)

    def _format_conflicts(self, conflicts):
        return MAPGDUtils.format_conflicts(conflicts)


class BanditBasedSelector:
    def __init__(self, config):
        self.config = config
        self.selection_strategy = config.get('selection_strategy', 'ucb')
        self.evaluation_budget = config.get('evaluation_budget', 50)
        self.min_evaluations = config.get('min_evaluations_per_candidate', 3)
        self.async_evaluation = config.get('async_evaluation', True)
        self.max_concurrent_evaluations = config.get('max_concurrent_evaluations', 4)

    def evaluate_prompts(self, candidate_prompts, training_data, task, predictor):
        if not candidate_prompts:
            return []
        print(f"Evaluating {len(candidate_prompts)} candidate prompts...")
        scores = [0.0] * len(candidate_prompts)
        eval_sample_size = min(150, len(training_data))
        eval_data = random.sample(training_data, eval_sample_size)

        print(f"  Using {eval_sample_size} samples for evaluation")

        max_workers = EXPERIMENT_CONFIG.get('max_concurrent_evaluators', 4)
        print(f"  Running up to {max_workers} concurrent evaluations")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._eval_single, candidate, eval_data, task, predictor, eval_sample_size): idx
                for idx, candidate in enumerate(candidate_prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    score = future.result()
                    print(f"    Candidate {idx + 1}/{len(candidate_prompts)} Score: {score:.3f}")
                except Exception as e:
                    print(f"    Candidate {idx + 1} Failed: {e}")
                    score = 0.0
                scores[idx] = score

        print(f"Evaluation completed. Scores: {[f'{s:.3f}' for s in scores]}")
        return scores

    def _eval_single(self, candidate, eval_data, task, predictor, n):
        score, _, _, _ = task.evaluate_prompt(candidate, eval_data, predictor, n=n)
        return score

    def select_top_candidates(self, candidate_prompts, task, training_data):
        if not candidate_prompts:
            return []

        if self.selection_strategy == 'ucb':
            return self._ucb_selection(candidate_prompts, task, training_data)
        elif self.selection_strategy == 'thompson':
            return self._thompson_sampling(candidate_prompts, task, training_data)
        else:
            return self._epsilon_greedy_selection(candidate_prompts, task, training_data)

    def _ucb_selection(self, candidates, task, training_data):
        print(f"UCB Selection: evaluating {len(candidates)} candidates")

        c = self.config.get('c', 1.0)
        num_rounds = min(20, self.evaluation_budget // len(candidates))
        samples_per_eval = min(10, len(training_data) // 4)

        print(f"  UCB rounds: {num_rounds}, samples per eval: {samples_per_eval}")

        counts = np.zeros(len(candidates))
        scores = np.zeros(len(candidates))

        for round_idx in range(num_rounds):
            print(f"  Round {round_idx + 1}/{num_rounds}")

            if np.sum(counts) == 0:
                selected_indices = list(range(len(candidates)))
            else:
                avg_scores = np.divide(scores, counts + 1e-3, out=np.zeros_like(scores), where=counts > 0)
                confidence = c * np.sqrt(np.log(round_idx + 1) / (counts + 1e-3))
                ucb_values = avg_scores + confidence

                k = min(len(candidates), 3)
                selected_indices = np.argsort(ucb_values)[::-1][:k]

            eval_data = random.sample(training_data, samples_per_eval)

            for idx in selected_indices:
                try:
                    score, _, _, _ = task.evaluate_prompt(candidates[idx], eval_data, None, n=samples_per_eval)

                    counts[idx] += samples_per_eval
                    scores[idx] += score * samples_per_eval

                    avg_score = scores[idx] / counts[idx] if counts[idx] > 0 else 0
                    print(f"    Candidate {idx}: score={score:.3f}, avg={avg_score:.3f}")

                except Exception as e:
                    print(f"    Candidate {idx}: evaluation failed ({e})")
                    counts[idx] += 1

        final_scores = np.divide(scores, counts, out=np.zeros_like(scores), where=counts > 0)

        beam_size = self.config.get('beam_size', 3)
        top_indices = np.argsort(final_scores)[::-1][:beam_size]
        selected_candidates = [candidates[i] for i in top_indices]

        print(f"  UCB final scores: {[f'{final_scores[i]:.3f}' for i in top_indices]}")
        print(f"  Selected {len(selected_candidates)} candidates")

        return selected_candidates

    def _thompson_sampling(self, candidates, task, training_data):
        print(f"Thompson Sampling: evaluating {len(candidates)} candidates")

        candidate_scores = []
        eval_sample_size = min(10, len(training_data))
        eval_data = random.sample(training_data, eval_sample_size)

        for i, candidate in enumerate(candidates):
            try:
                score, _, _, _ = task.evaluate_prompt(candidate, eval_data, None, n=eval_sample_size)

                alpha = 1 + score * 10
                beta = 1 + (1 - score) * 10
                sampled_score = np.random.beta(alpha, beta)

                candidate_scores.append((sampled_score, candidate))
                print(f"  Candidate {i + 1}: score={score:.3f}, sampled={sampled_score:.3f}")

            except Exception as e:
                print(f"  Candidate {i + 1}: failed ({e})")
                candidate_scores.append((0.0, candidate))

        candidate_scores.sort(reverse=True, key=lambda x: x[0])
        beam_size = self.config.get('beam_size', 3)
        return [candidate for _, candidate in candidate_scores[:beam_size]]

    def _epsilon_greedy_selection(self, candidates, task, training_data):
        epsilon = 0.1
        print(f"Epsilon-Greedy (ε={epsilon}): evaluating {len(candidates)} candidates")

        beam_size = self.config.get('beam_size', 3)

        if random.random() < epsilon:
            print("  Using exploration (random selection)")
            return random.sample(candidates, min(beam_size, len(candidates)))
        else:
            print("  Using exploitation (greedy selection)")
            candidate_scores = []
            eval_sample_size = min(8, len(training_data))
            eval_data = random.sample(training_data, eval_sample_size)

            for i, candidate in enumerate(candidates):
                try:
                    score, _, _, _ = task.evaluate_prompt(candidate, eval_data, None, n=eval_sample_size)
                    candidate_scores.append((score, candidate))
                    print(f"  Candidate {i + 1}: score={score:.3f}")
                except Exception as e:
                    print(f"  Candidate {i + 1}: failed ({e})")
                    candidate_scores.append((0.0, candidate))

            candidate_scores.sort(reverse=True, key=lambda x: x[0])
            return [candidate for _, candidate in candidate_scores[:beam_size]]


class AgentManager:
    def __init__(self, config):
        self.config = config
        self.num_agents = config.get('num_agents', 4)
        self.task_name = config.get('task_name', 'default')
        print(f"🔍 AgentManager initialized for task: {self.task_name}")

        self.synchronization_strategy = config.get('sync_strategy', 'best_prompt_sharing')
        self.agents = []
        self.async_mode = config.get('async_mode', True)
        self.max_concurrent_agents = config.get('max_concurrent_agents', 4)

    def initialize_agents(self, initial_prompt):
        available_agent_types = SpecializedPromptAgent.get_available_agent_types(self.task_name)

        random_roles = self.config.get("random_agent_roles", False)
        mode = "random" if random_roles else "specialized"

        print(f"Initializing {self.num_agents} {mode} agents for task '{self.task_name}'...")
        print(f"Available agent types: {available_agent_types}")

        for i in range(self.num_agents):
            if random_roles:
                agent_type = random.choice(available_agent_types)
            else:
                agent_type = available_agent_types[i % len(available_agent_types)]

            agent = SpecializedPromptAgent(agent_type, task_name=self.task_name)
            agent.current_prompt = initial_prompt
            self.agents.append(agent)

            print(f"  Agent {i + 1}: {agent_type} - {agent.config['role_description']}")

        print(
            f"Agent initialization complete for {self.task_name}. Ready for {'async' if self.async_mode else 'sync'} optimization."
        )

    def get_reproducibility_manifest(self):
        return {
            'task_name': self.task_name,
            'num_agents': len(self.agents),
            'agent_sequence': [agent.agent_type for agent in self.agents],
            'agent_prompts': [agent.get_reproducibility_manifest() for agent in self.agents],
        }

    async def generate_gradients_async(self, training_data, task, predictor):
        print(f"Starting gradient generation from {len(self.agents)} agents...")
        start_time = time.time()
        loop = asyncio.get_event_loop()
        shared_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_agents)

        async def generate_single_agent_gradient(agent_idx, agent):
            try:
                print(f"  Agent {agent_idx + 1} ({agent.agent_type}): starting analysis...")

                sample_size = min(20, len(training_data))
                start_idx = (agent_idx * 10) % len(training_data)
                end_idx = min(start_idx + sample_size, len(training_data))

                if end_idx - start_idx < sample_size:
                    sample_data = random.sample(training_data, min(sample_size, len(training_data)))
                else:
                    sample_data = training_data[start_idx:end_idx]

                score, texts, labels, preds = await loop.run_in_executor(
                    shared_executor,
                    lambda: task.evaluate_prompt(agent.current_prompt, sample_data, predictor, n=len(sample_data)),
                )

                error_examples = []
                for text, label, pred in zip(texts, labels, preds):
                    if label != pred:
                        error_examples.append({'text': text, 'true_label': label, 'predicted_label': pred})
                gradients = await agent.generate_specialized_gradient_async(agent.current_prompt, error_examples, task)

                print(f"  Agent {agent_idx + 1} completed: {len(gradients)} gradients")
                return f'agent_{agent_idx}', gradients

            except Exception as e:
                print(f"  Agent {agent_idx + 1} failed: {e}")
                return f'agent_{agent_idx}', [f"Improve the prompt for better {agent.config['focus']}"]

        tasks = [generate_single_agent_gradient(i, agent) for i, agent in enumerate(self.agents)]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        shared_executor.shutdown(wait=True)

        agent_gradients = {}
        successful_agents = 0

        for result in results:
            if isinstance(result, tuple):
                agent_id, gradients = result
                agent_gradients[agent_id] = gradients
                successful_agents += 1
            else:
                print(f"  Task failed with exception: {result}")

        total_time = time.time() - start_time
        print('gradient generation completed:')
        print(f"  - {successful_agents}/{len(self.agents)} agents successful")
        print(f"  - Total time: {total_time:.2f}s")
        print(f"  - Average time per agent: {total_time / len(self.agents):.2f}s")

        return agent_gradients

    def generate_gradients(self, training_data, task, predictor):
        if self.async_mode:
            print("Using ASYNC mode for gradient generation...")
            return asyncio.run(self.generate_gradients_async(training_data, task, predictor))
        else:
            print("Using SYNC mode for gradient generation...")
            return self._generate_gradients_sync(training_data, task, predictor)

    def _generate_gradients_sync(self, training_data, task, predictor):
        agent_gradients = {}

        print(f"Generating specialized gradients from {len(self.agents)} agents...")

        for i, agent in enumerate(self.agents):
            print(f"  Agent {i + 1} ({agent.agent_type}): analyzing prompt performance...")

            sample_size = min(20, len(training_data))
            sample_data = random.sample(training_data, sample_size)

            try:
                score, texts, labels, preds = task.evaluate_prompt(
                    agent.current_prompt, sample_data, predictor, n=len(sample_data)
                )

                error_examples = []
                for text, label, pred in zip(texts, labels, preds):
                    if label != pred:
                        error_examples.append({'text': text, 'true_label': label, 'predicted_label': pred})

                if not error_examples:
                    error_examples = [
                        {
                            'text': ex.get('text', ''),
                            'true_label': ex.get('label', ''),
                            'predicted_label': ex.get('label', ''),
                        }
                        for ex in sample_data[:3]
                    ]

                gradients = agent.generate_specialized_gradient(agent.current_prompt, error_examples, task)

                agent_gradients[f'agent_{i}'] = gradients

            except Exception as e:
                print(f"Error generating gradients for agent {i}: {e}")
                agent_gradients[f'agent_{i}'] = [f"Improve the prompt for better {agent.config['focus']}"]

        return agent_gradients

    def get_current_prompts(self):
        return [agent.current_prompt for agent in self.agents]

    def update_agents(self, selected_prompts):
        for i, agent in enumerate(self.agents):
            if i < len(selected_prompts):
                agent.current_prompt = selected_prompts[i]


class CollaborativePromptExpander:
    def __init__(self, config):
        self.beam_size = config.get('beam_size', 3)
        self.mc_samples = config.get('mc_samples', 2)
        self.successor_candidates = config.get('successor_candidates', max(4, self.beam_size * 2))
        self.diversity_threshold = config.get('diversity_threshold', 0.7)
        self.expansion_strategy = config.get('expansion_strategy', 'beam_search')

        self.semantic_model = MAPGDUtils.get_semantic_model()
        if self.semantic_model:
            print('CollaborativePromptExpander: Using shared semantic model for diversity filtering')
        else:
            print('Warning: No semantic model available for diversity filtering')

    def get_reproducibility_manifest(self):
        return {
            'gradient_application_system_prompt': (
                "You are a professional expert in prompt word optimization, "
                "skilled in applying semantic gradients for precise improvements."
            ),
            'gradient_application_user_template': """
I'm trying to write a zero-shot classifier.
My current prompt is:
"{prompt}"
the gradient of this prompt is {gradient}
Based on the above information, I wrote{steps_per_gradient} different improved prompts.
The {steps_per_gradient} new prompts are wrapped with <START> and <END>:
""".strip(),
            'stochastic_variant_user_template': """
Generate a variation of the following instruction while keeping the semantic meaning.

Input: {prompt}
Each output prompt is wrapped with <START> and <END>.
""".strip(),
        }

    def expand(self, current_prompt, gradients):
        print(f"Expanding prompts with strategy: {self.expansion_strategy}")

        candidates = []
        debug_summary = {
            'strategy': self.expansion_strategy,
            'num_gradients': len(gradients) if gradients else 0,
            'beam_raw': 0,
            'mc_raw': 0,
            'combined_raw': 0,
            'sanitized': 0,
            'diverse': 0,
        }

        if self.expansion_strategy == 'beam_search':
            candidates = self._beam_search_expansion(current_prompt, gradients)
            print(f"Beam search strategy generated {len(candidates)} candidates")

        elif self.expansion_strategy == 'monte_carlo':
            candidates = self._monte_carlo_paraphrasing(current_prompt)
            debug_summary['mc_raw'] = len(candidates)
            print(f"Monte Carlo strategy generated {len(candidates)} variants")
        else:
            beam_candidates = self._beam_search_expansion(current_prompt, gradients)
            mc_candidates = self._monte_carlo_paraphrasing(current_prompt)
            debug_summary['beam_raw'] = len(beam_candidates)
            debug_summary['mc_raw'] = len(mc_candidates)
            candidates = beam_candidates + mc_candidates
            print(f"Hybrid strategy generated {len(candidates)} candidates")

        debug_summary['combined_raw'] = len(candidates)
        candidates = self._sanitize_candidates(candidates, current_prompt)
        debug_summary['sanitized'] = len(candidates)

        diverse_candidates = self._filter_for_diversity(candidates)
        debug_summary['diverse'] = len(diverse_candidates)
        print(f"Filtered to {len(diverse_candidates)} diverse candidates")
        print(
            "[CandidateDebug] "
            f"strategy={debug_summary['strategy']} "
            f"gradients={debug_summary['num_gradients']} "
            f"beam_raw={debug_summary['beam_raw']} "
            f"mc_raw={debug_summary['mc_raw']} "
            f"combined_raw={debug_summary['combined_raw']} "
            f"sanitized={debug_summary['sanitized']} "
            f"diverse={debug_summary['diverse']}"
        )
        if candidates:
            print(f"[CandidateDebug] sanitized_preview={repr(candidates[:2])}")
        if diverse_candidates:
            print(f"[CandidateDebug] diverse_preview={repr(diverse_candidates[:2])}")

        return diverse_candidates[: self.beam_size * 2]

    def _beam_search_expansion(self, current_prompt, gradients):
        beam_candidates = []
        if not gradients:
            print("[CandidateDebug] beam_search skipped: no gradients, falling back to current prompt only")
            return [current_prompt]

        steps_per_gradient = max(2, int(np.ceil(self.successor_candidates / max(1, len(gradients)))))
        print(
            "[CandidateDebug] beam_search "
            f"gradients={len(gradients)} "
            f"successor_candidates={self.successor_candidates} "
            f"steps_per_gradient={steps_per_gradient}"
        )

        for idx, gradient in enumerate(gradients):
            expanded_prompt = self._apply_gradient(current_prompt, gradient, steps_per_gradient=steps_per_gradient)
            print(
                "[CandidateDebug] beam_gradient "
                f"index={idx} generated={len(expanded_prompt)} "
                f"gradient_preview={repr(str(gradient)[:120])}"
            )
            beam_candidates.extend(expanded_prompt)

        if current_prompt not in beam_candidates:
            beam_candidates.append(current_prompt)
            print("[CandidateDebug] beam_search appended current prompt as fallback")
        return beam_candidates

    def _monte_carlo_paraphrasing(self, base_prompt):
        mc_variants = []

        for sample_idx in range(self.mc_samples):
            variant = self._generate_stochastic_variant(base_prompt)
            print("[CandidateDebug] monte_carlo " f"sample={sample_idx} generated={len(variant)}")
            if variant:
                mc_variants.extend(variant)

        return mc_variants

    def _apply_gradient(self, prompt, gradient, steps_per_gradient=1):
        application_prompt = f"""
        I'm trying to write a zero-shot classifier.
        My current prompt is:
        "{prompt}"
        the gradient of this prompt is {gradient}
        Keep the same task and output label semantics.
        Preserve any valid placeholders already used by the prompt, especially {{text}}.
        Do not invent new placeholders unless they already appear in the current prompt.
        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        The {steps_per_gradient} new prompts are wrapped with <START> and <END>:
        """

        response = call_openai(
            prompt=application_prompt,
            system_prompt="You are a professional expert in prompt word optimization, skilled in applying semantic gradients for precise improvements.",
        )

        pattern = r'<\s*START\s*>(.*?)<\s*END\s*>'
        candidates = MAPGDUtils.parse_llm_response_with_tags(response, pattern)
        if not re.search(pattern, response, re.DOTALL | re.IGNORECASE):
            print("[CandidateDebug] apply_gradient dropped response: missing <START>/<END> tags")
            return []
        print(f"[CandidateDebug] apply_gradient accepted {len(candidates)} tagged candidates")
        return candidates

    def _sanitize_candidates(self, candidates, current_prompt):
        sanitized = []
        allowed_placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", current_prompt))
        current_has_yes_no = 'yes' in current_prompt.lower() and 'no' in current_prompt.lower()
        for candidate in candidates:
            candidate_text = MAPGDUtils.extract_gradient_text(candidate).strip()
            if not candidate_text:
                print("[CandidateDebug] sanitize_drop reason=empty")
                continue
            if len(candidate_text) < 80:
                print(
                    f"[CandidateDebug] sanitize_drop reason=too_short chars={len(candidate_text)} text={repr(candidate_text[:80])}"
                )
                continue
            if len(candidate_text.split()) < 12:
                print(
                    f"[CandidateDebug] sanitize_drop reason=too_few_words words={len(candidate_text.split())} text={repr(candidate_text[:80])}"
                )
                continue
            candidate_placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", candidate_text))
            if candidate_placeholders - allowed_placeholders:
                print(
                    "[CandidateDebug] sanitize_drop "
                    f"reason=new_placeholders found={sorted(candidate_placeholders - allowed_placeholders)}"
                )
                continue
            if allowed_placeholders and '{text}' in current_prompt and '{text}' not in candidate_text:
                print("[CandidateDebug] sanitize_drop reason=missing_text_placeholder")
                continue
            lower_candidate = candidate_text.lower()
            if current_has_yes_no and not ('yes' in lower_candidate and 'no' in lower_candidate):
                print("[CandidateDebug] sanitize_drop reason=missing_yes_no_semantics")
                continue
            if 'cannot determine' in lower_candidate:
                print("[CandidateDebug] sanitize_drop reason=contains_cannot_determine")
                continue
            if re.fullmatch(r"[a-zA-Z\s,.;:!?'\"]+", candidate_text) and len(set(candidate_text.lower().split())) <= 3:
                print(f"[CandidateDebug] sanitize_drop reason=fragment text={repr(candidate_text[:80])}")
                continue
            sanitized.append(candidate_text)
        if current_prompt not in sanitized:
            sanitized.append(current_prompt)
            print("[CandidateDebug] sanitize appended current prompt as fallback")
        return sanitized

    def _generate_stochastic_variant(self, prompt):
        variant_prompt = f"""
        Generate a variation of the following instruction while keeping the semantic meaning.

        Input: {prompt}
        Each output prompt is wrapped with <START> and <END>.
        """

        response = call_openai(prompt=variant_prompt, system_prompt="")

        pattern = r'<\s*START\s*>(.*?)<\s*END\s*>'
        variants = MAPGDUtils.parse_llm_response_with_tags(response, pattern)
        if not re.search(pattern, response, re.DOTALL | re.IGNORECASE):
            print("[CandidateDebug] monte_carlo dropped response: missing <START>/<END> tags")
            return []
        print(f"[CandidateDebug] monte_carlo accepted {len(variants)} tagged variants")
        return variants

    def _filter_for_diversity(self, candidates):
        return MAPGDUtils.filter_for_diversity(
            candidates, diversity_threshold=self.diversity_threshold, max_candidates=self.beam_size * 2
        )


class ConvergenceMonitor:
    def __init__(self, config):
        self.convergence_threshold = config.get('convergence_threshold', 0.01)
        self.patience = config.get('patience', 3)
        self.performance_history = []
        self.no_improvement_count = 0

    def check_convergence(self, current_performance=None):
        if current_performance is not None:
            self.performance_history.append(current_performance)

        if len(self.performance_history) < 2:
            return False

        recent_improvement = self.performance_history[-1] - self.performance_history[-2]

        if recent_improvement < self.convergence_threshold:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0

        return self.no_improvement_count >= self.patience
