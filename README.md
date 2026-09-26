# MAPGD Quick Start

The remaining code is being organized and will be released in future updates.

Run these commands from the project root in Windows PowerShell.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-api-key"
$env:PYTHONIOENCODING = "utf-8"
```

Keep the embedding model in `all-MiniLM-L6-v2/`. The default API model is
`gpt-4o-mini`.

## Run selected datasets

Run one dataset:

```powershell
.\.venv\Scripts\python.exe experiment_baseline.py --task [your task]
```

Available tasks: `liar`, `ethos`, `jailbreak`, `gsm8k`, `aqua`, `svamp`, and
`sarcasm`. Data is loaded from `data/`.

Run several datasets sequentially:

```powershell
foreach ($task in @("liar", "ethos", "gsm8k")) {
    .\.venv\Scripts\python.exe experiment_baseline.py --task $task
}
```

## Small trial run

Temporarily reduce iterations and sample sizes for LIAR:

```powershell
@'
from liar_config import EXPERIMENT_CONFIG
from experiment_baseline import run_baseline_experiment

EXPERIMENT_CONFIG.update(
    max_iterations=1, minibatch_size=8, dev_size=8, test_size=10,
    beam_size=1, evaluation_budget=8, successor_candidates=2,
    monte_carlo_samples=0,
)
if run_baseline_experiment(task="liar") is None:
    raise SystemExit("Experiment failed; check the output above.")
'@ | .\.venv\Scripts\python.exe -
```

This loads the full files but uses smaller samples for optimization and
evaluation. It calls the model API and leaves configuration files unchanged.


## Results

Successful runs save `baseline_<task>_results_<timestamp>.json` in the current
directory, including the configuration, best prompt, optimization history, and
final test score (F1 for classification; accuracy for math tasks).

## Acknowledgments

We thank the authors of ProTeGi for their pioneering work on automatic prompt optimization and for sharing their open-source implementation.
