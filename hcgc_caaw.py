import re
import time

import numpy as np
from sklearn.cluster import KMeans

from llm import call_openai
from utils import MAPGDUtils


class HypersphereConstrainedGradientClustering:
    def __init__(self, config):
        self.conflict_threshold = config.get('conflict_threshold', 0.3)
        self.max_clusters = config.get('max_clusters', 5)
        self.margin_scale = config.get('margin_scale', 2.0)
        self.temperature = config.get('clustering_temperature', 0.1)
        self.conflict_log_path = config.get('conflict_log_path', 'hcgc_conflicts.log')
        self.enable_directional_conflict_check = config.get('directional_conflict_check', True)
        self.last_pair_analysis = []
        self.last_complementary_pairs = []

        self.component_keywords = {
            'instruction': {'instruction', 'instructions', 'guideline', 'guidelines', 'task', 'rule', 'rules'},
            'example': {'example', 'examples', 'few-shot', 'shot', 'demonstration', 'demonstrations'},
            'format': {'format', 'template', 'templates', 'structure', 'structured', 'label', 'labels', 'output'},
            'style': {'style', 'tone', 'wording', 'language', 'phrasing'},
            'reasoning': {'reasoning', 'logic', 'step', 'steps', 'decompose', 'decomposition', 'chain'},
            'calculation': {'calculate', 'calculation', 'compute', 'computation', 'numerical', 'arithmetic', 'answer'},
            'context': {'context', 'background', 'metadata', 'evidence', 'information'},
            'constraint': {'constraint', 'constraints', 'criterion', 'criteria', 'condition', 'conditions'},
        }
        self.polarity_keywords = {
            'increase': {
                'add',
                'include',
                'introduce',
                'increase',
                'expand',
                'emphasize',
                'strengthen',
                'clarify',
                'specify',
                'detail',
                'explain',
                'highlight',
            },
            'decrease': {
                'remove',
                'omit',
                'drop',
                'decrease',
                'reduce',
                'downplay',
                'simplify',
                'shorten',
                'avoid',
                'eliminate',
            },
        }

        self.semantic_model = MAPGDUtils.get_semantic_model()

    def apply(self, agent_gradients):
        print("[HCGC] Starting hypersphere-constrained clustering...")

        gradient_vectors, gradient_metadata = self._embed_to_hypersphere(agent_gradients)
        if len(gradient_vectors) == 0:
            print("[HCGC] No gradients to cluster")
            return []

        conflicts = self._detect_conflicts(gradient_vectors, gradient_metadata)
        self._log_conflicts(conflicts, gradient_metadata)
        print(f"[HCGC] Detected {len(conflicts)} gradient conflicts")

        n_clusters = min(len(gradient_vectors), self.max_clusters)
        cluster_labels, centroids = self._cluster_gradients(gradient_vectors, n_clusters)
        print(f"[HCGC] Formed {n_clusters} semantic clusters")

        cluster_labels = self._apply_angular_margin(gradient_vectors, cluster_labels, centroids)
        clusters = self._organize_clusters(gradient_metadata, cluster_labels, centroids)
        fused_gradients = self._fuse_clusters(clusters, conflicts)

        print(f"[HCGC] Completed clustering: {len(fused_gradients)} fused gradients")
        return fused_gradients

    def _embed_to_hypersphere(self, agent_gradients):
        all_texts = []
        metadata = []

        for agent_id, gradients in agent_gradients.items():
            for idx, gradient in enumerate(gradients):
                gradient_text = MAPGDUtils.extract_gradient_text(gradient)
                all_texts.append(gradient_text)
                metadata.append(
                    {
                        'agent_id': agent_id,
                        'gradient_idx': idx,
                        'gradient_text': gradient_text,
                        'signature': self._extract_refinement_signature(gradient_text),
                    }
                )

        vectors = MAPGDUtils.encode_texts(all_texts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_vectors = vectors / norms
        return normalized_vectors, metadata

    def _detect_conflicts(self, normalized_vectors, metadata):
        conflicts = []
        complementary_pairs = []
        pair_analysis = []
        n = len(normalized_vectors)

        for i in range(n):
            for j in range(i + 1, n):
                similarity = np.dot(normalized_vectors[i], normalized_vectors[j])
                if similarity < self.conflict_threshold:
                    pair_record = {
                        'idx_i': i,
                        'idx_j': j,
                        'agent_i': metadata[i]['agent_id'],
                        'agent_j': metadata[j]['agent_id'],
                        'gradient_i': metadata[i].get('gradient_text', ''),
                        'gradient_j': metadata[j].get('gradient_text', ''),
                        'similarity': float(similarity),
                        'angle': float(np.arccos(np.clip(similarity, -1, 1))),
                    }
                    pair_record.update(self._analyze_pair_relation(metadata[i], metadata[j]))
                    pair_analysis.append(pair_record)

                    if pair_record['relation'] == 'conflict':
                        conflicts.append(pair_record)
                    else:
                        complementary_pairs.append(pair_record)

        self.last_pair_analysis = pair_analysis
        self.last_complementary_pairs = complementary_pairs
        return conflicts

    def _analyze_pair_relation(self, grad_i, grad_j):
        if not self.enable_directional_conflict_check:
            return {
                'relation': 'conflict',
                'screening_reason': 'directional_check_disabled',
            }

        signature_i = grad_i.get('signature') or self._extract_refinement_signature(grad_i.get('gradient_text', ''))
        signature_j = grad_j.get('signature') or self._extract_refinement_signature(grad_j.get('gradient_text', ''))
        shared_components = sorted(signature_i['components'] & signature_j['components'])
        polarity_i = signature_i['polarity']
        polarity_j = signature_j['polarity']
        opposing_polarity = polarity_i != 'neutral' and polarity_j != 'neutral' and polarity_i != polarity_j

        if shared_components and opposing_polarity:
            return {
                'relation': 'conflict',
                'screening_reason': 'shared_component_with_opposing_polarity',
                'shared_components': shared_components,
                'polarity_i': polarity_i,
                'polarity_j': polarity_j,
            }

        return {
            'relation': 'complementary',
            'screening_reason': 'low_similarity_without_directional_opposition',
            'shared_components': shared_components,
            'polarity_i': polarity_i,
            'polarity_j': polarity_j,
        }

    def _extract_refinement_signature(self, gradient_text):
        tokens = set(re.findall(r"[a-zA-Z][a-zA-Z\-]*", gradient_text.lower()))
        components = {component for component, keywords in self.component_keywords.items() if tokens & keywords}

        increase_hits = len(tokens & self.polarity_keywords['increase'])
        decrease_hits = len(tokens & self.polarity_keywords['decrease'])
        if increase_hits > decrease_hits:
            polarity = 'increase'
        elif decrease_hits > increase_hits:
            polarity = 'decrease'
        else:
            polarity = 'neutral'

        return {
            'components': components,
            'polarity': polarity,
        }

    def _log_conflicts(self, conflicts, metadata):
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.conflict_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[{timestamp}] conflict_threshold={self.conflict_threshold}, "
                    f"detected={len(conflicts)}, complementary={len(self.last_complementary_pairs)}\n"
                )
                for c in conflicts:
                    grad_i = metadata[c['idx_i']].get('gradient_text', '')
                    grad_j = metadata[c['idx_j']].get('gradient_text', '')
                    f.write(
                        f"  - agents: {c['agent_i']} vs {c['agent_j']}, "
                        f"indices: {c['idx_i']} vs {c['idx_j']}, "
                        f"similarity={c['similarity']:.4f}, angle={c['angle']:.4f}\n"
                    )
                    f.write(f"    gradient_{c['idx_i']} ({c['agent_i']}): {grad_i}\n")
                    f.write(f"    gradient_{c['idx_j']} ({c['agent_j']}): {grad_j}\n")
                    if c.get('shared_components'):
                        f.write(f"    shared_components: {', '.join(c['shared_components'])}\n")
                    f.write(
                        f"    directional_check: {c.get('polarity_i', 'neutral')} "
                        f"vs {c.get('polarity_j', 'neutral')} ({c.get('screening_reason', 'n/a')})\n"
                    )

                if self.last_complementary_pairs:
                    for c in self.last_complementary_pairs:
                        f.write(
                            f"  ~ agents: {c['agent_i']} vs {c['agent_j']}, "
                            f"similarity={c['similarity']:.4f}, reason={c.get('screening_reason', 'n/a')}\n"
                        )
                        if c.get('shared_components'):
                            f.write(f"    shared_components: {', '.join(c['shared_components'])}\n")
                f.write("\n")
        except Exception as e:
            print(f"[HCGC] Failed to write conflict log: {e}")

    def _cluster_gradients(self, normalized_vectors, n_clusters):
        if n_clusters <= 1:
            return np.zeros(len(normalized_vectors), dtype=int), normalized_vectors.mean(axis=0, keepdims=True)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_vectors)
        centroids = kmeans.cluster_centers_
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.where(centroid_norms == 0, 1, centroid_norms)
        return labels, centroids

    def _apply_angular_margin(self, normalized_vectors, cluster_labels, centroids):
        n = self.margin_scale
        reassignments = 0

        for i, vec in enumerate(normalized_vectors):
            current_cluster = cluster_labels[i]
            similarities = np.dot(centroids, vec)
            angles = np.arccos(np.clip(similarities, -1, 1))
            alpha = angles[current_cluster]

            violates_margin = False
            for j, beta in enumerate(angles):
                if j == current_cluster:
                    continue
                if n * alpha >= beta:
                    violates_margin = True
                    break

            if violates_margin:
                best_cluster = current_cluster
                best_score = -np.inf
                for j in range(len(centroids)):
                    alpha_j = angles[j]
                    satisfies = all(n * alpha_j < angles[k] for k in range(len(centroids)) if k != j)
                    if satisfies and similarities[j] > best_score:
                        best_score = similarities[j]
                        best_cluster = j

                if best_cluster != current_cluster:
                    cluster_labels[i] = best_cluster
                    reassignments += 1

        if reassignments > 0:
            print(f"[HCGC] Applied angular margin: {reassignments} reassignments")
        return cluster_labels

    def _organize_clusters(self, metadata, cluster_labels, centroids):
        clusters = []
        n_clusters = len(centroids)

        for cluster_id in range(n_clusters):
            cluster_gradients = []
            cluster_agents = set()

            for i, label in enumerate(cluster_labels):
                if label == cluster_id:
                    cluster_gradients.append(metadata[i]['gradient_text'])
                    cluster_agents.add(metadata[i]['agent_id'])

            if cluster_gradients:
                clusters.append(
                    {
                        'id': cluster_id,
                        'gradients': cluster_gradients,
                        'agents': list(cluster_agents),
                        'centroid': centroids[cluster_id],
                    }
                )

        return clusters

    def _fuse_clusters(self, clusters, conflicts):
        fused_gradients = []

        for cluster in clusters:
            if len(cluster['gradients']) == 1:
                fused_gradients.append(cluster['gradients'][0])
            else:
                fusion_prompt = self._create_fusion_prompt(cluster['gradients'], conflicts)
                response = call_openai(
                    prompt=fusion_prompt,
                    system_prompt=(
                        "You are an expert at synthesizing multiple improvement suggestions "
                        "into coherent unified recommendations."
                    ),
                )
                fused_gradients.append(MAPGDUtils.parse_fusion_response(response))

        return fused_gradients

    def _create_fusion_prompt(self, gradients, conflicts, num_outputs=1):
        gradient_texts = "\n".join(f"Suggestion {i + 1}: {grad}" for i, grad in enumerate(gradients))

        conflict_info = ""
        if conflicts:
            conflict_info = (
                "\n\nNote: Some suggestions are directionally conflicting. "
                "Resolve only explicit contradictions; keep complementary refinements when possible."
            )

        return f"""
I need to combine the following {len(gradients)} prompt improvement suggestions into {num_outputs} unified, coherent improvement(s):

{gradient_texts}{conflict_info}

Please synthesize these into a single, actionable improvement that captures the best aspects of all suggestions while resolving any conflicts.

Wrap the unified improvement with <START> and <END>
"""


class ChannelAdaptiveAgentWeighting:
    def __init__(self, config):
        self.lambda_param = config.get('caaw_lambda', 1.0)
        self.validation_samples = config.get('caaw_validation_samples', 20)
        self.enable_weighting = config.get('enable_caaw', True)
        self.burn_in_iterations = config.get('caaw_burn_in_iterations', 2)
        self.last_fusion_records = []
        self.last_fusion_agents = set()
        self.last_weighting_mode = 'uniform'
        self.last_weighting_reason = 'initial_state'

    def apply(self, clustered_gradients, agent_gradients, validation_data, task, predictor):
        if not self.enable_weighting:
            print("[CAAW] Disabled - using uniform weighting")
            return clustered_gradients

        print("[CAAW] Applying channel-adaptive weighting...")
        weighted_gradients = list(clustered_gradients)
        print(f"[CAAW] Completed weighting: {len(weighted_gradients)} gradients")
        return weighted_gradients

    def compute_agent_weights(self, agent_ids, agent_history, current_iteration):
        if current_iteration < self.burn_in_iterations:
            self.last_weighting_mode = 'uniform'
            self.last_weighting_reason = 'burn_in'
            return self._uniform_weights(agent_ids)

        if not agent_history or len(agent_history) < 2:
            self.last_weighting_mode = 'uniform'
            self.last_weighting_reason = 'insufficient_history'
            return self._uniform_weights(agent_ids)

        filtered_history = [h for h in agent_history if h.get('agent_id') in agent_ids and h.get('accepted', True)]
        if not filtered_history:
            self.last_weighting_mode = 'uniform'
            self.last_weighting_reason = 'no_accepted_history'
            return self._uniform_weights(agent_ids)

        gains = {}
        for agent_id in agent_ids:
            agent_gains = [h.get('improvement', 0.0) for h in filtered_history if h.get('agent_id') == agent_id]
            gains[agent_id] = float(np.mean(agent_gains[-3:])) if agent_gains else 0.0

        gain_values = np.array([gains[aid] for aid in agent_ids])
        exp_gains = np.exp(self.lambda_param * gain_values)
        denom = exp_gains.sum()
        if denom == 0:
            weights_array = np.ones_like(exp_gains) / len(exp_gains)
        else:
            weights_array = exp_gains / denom

        weights = {agent_id: float(w) for agent_id, w in zip(agent_ids, weights_array)}
        self.last_weighting_mode = 'adaptive'
        self.last_weighting_reason = 'historical_validation_gain'
        print(f"[CAAW] Agent weights: {weights}")
        return weights

    def _uniform_weights(self, agent_ids):
        if not agent_ids:
            return {}
        uniform = 1.0 / len(agent_ids)
        return {agent_id: uniform for agent_id in agent_ids}

    def weighted_fusion(self, gradients, agent_weights, agent_to_gradient_map):
        if len(gradients) == 1:
            return gradients[0]

        weighted_items = []
        for i, grad_text in enumerate(gradients):
            agent_id = None
            for aid, grad_list in agent_to_gradient_map.items():
                if grad_text in grad_list:
                    agent_id = aid
                    break

            weight = agent_weights.get(agent_id, 1.0 / max(1, len(agent_weights)))
            agent_label = agent_id if agent_id is not None else f"unknown_{i}"

            if weight > 0.4:
                emphasis = "strongly emphasize"
            elif weight > 0.2:
                emphasis = "moderately emphasize"
            else:
                emphasis = "slightly emphasize"

            weighted_items.append(f"Suggestion from {agent_label} (weight={weight:.2f}, {emphasis}): {grad_text}")

        weighted_text = "\n".join(weighted_items)
        fusion_prompt = f"""
Synthesize the following weighted improvement suggestions into a single, coherent improvement.
Pay close attention to the indicated weights and emphasis levels, giving more importance to suggestions with higher weights:

{weighted_text}

Wrap the final unified improvement with <START> and <END>
"""

        response = call_openai(
            prompt=fusion_prompt,
            system_prompt="You are an expert at weighted fusion of improvement suggestions.",
        )
        return MAPGDUtils.parse_fusion_response(response)


def integrate_hcgc_into_coordinator(coordinator, config):
    coordinator.hcgc = HypersphereConstrainedGradientClustering(config)
    original_coordinate = coordinator.coordinate_gradients

    def coordinate_gradients_with_hcgc(agent_gradients):
        if config.get('enable_hcgc', True):
            print("[Integration] Using HCGC for gradient coordination")
            return coordinator.hcgc.apply(agent_gradients)
        print("[Integration] Using original gradient coordination")
        return original_coordinate(agent_gradients)

    coordinator.coordinate_gradients = coordinate_gradients_with_hcgc
    return coordinator


def integrate_caaw_into_framework(framework, config):
    framework.caaw = ChannelAdaptiveAgentWeighting(config)
    framework.agent_performance_history = []
    return framework
