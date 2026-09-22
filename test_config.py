from init_prompt import TEST_INITIAL_PROMPT


DATA_CONFIG = {
    'task_name': 'test',
    'data_dir': '/mnt/d/agent/mapgd/data/test',
    'sentence_transformer_model': '/mnt/d/model/pretrained_model/all-MiniLM-L6-v2',
}


MAPGD_CONFIG = {
    'max_iterations': 10,
    'beam_size': 4,
    'num_agents': 4,
    'minibatch_size': 64,
    'error_group_size': 4,
    'gradients_per_group': 4,
    'monte_carlo_samples': 2,
    'successor_candidates': 8,
    'async_mode': True,
    'async_evaluation': True,
    'max_concurrent_agents': 4,
    'max_concurrent_evaluators': 4,
    'fusion_method': 'semantic_clustering',
    'conflict_threshold': 0.3,
    'cluster_threshold': 0.7,
    'max_clusters': 5,
    'expansion_strategy': 'beam_search',
    'mc_samples': 2,
    'diversity_threshold': 0.7,
    'use_gpu': True,
    'batch_size': 32,
    'max_workers': 4,
    'batch_encoding': True,
    'cache_enabled': True,
    'selection_strategy': 'ucb',
    'evaluation_budget': 80,
    'min_evaluations_per_candidate': 3,
    'c': 1.0,
    'epsilon': 0.1,
    'convergence_patience': 3,
    'convergence_threshold': 0.01,
}


EXPERIMENT_CONFIG = {**DATA_CONFIG, **MAPGD_CONFIG, 'initial_prompt': TEST_INITIAL_PROMPT}
