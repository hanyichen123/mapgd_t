from init_prompt import SARCASM_INITIAL_PROMPT
from paper_config import build_paper_config


EXPERIMENT_CONFIG = build_paper_config(
    task_name="sarcasm",
    data_subdir="sarcasm",
    initial_prompt=SARCASM_INITIAL_PROMPT,
)

DATA_CONFIG = {key: EXPERIMENT_CONFIG[key] for key in ("task_name", "data_dir", "sentence_transformer_model")}
