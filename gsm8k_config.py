from paper_config import build_paper_config


GSM8K_INITIAL_PROMPT = """
# Task
Solve the math word problem step by step.
# Instructions
1. Read the problem carefully
2. Identify what needs to be calculated
3. Show your work step by step
4. Provide the final numerical answer
# Output Format
Show your reasoning process and end with: #### [final_answer]
# Problem
{text}
""".strip()


EXPERIMENT_CONFIG = build_paper_config(
    task_name="gsm8k",
    data_subdir="gsm8k",
    initial_prompt=GSM8K_INITIAL_PROMPT,
)

DATA_CONFIG = {key: EXPERIMENT_CONFIG[key] for key in ("task_name", "data_dir", "sentence_transformer_model")}
