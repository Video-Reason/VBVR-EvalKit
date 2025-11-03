# VMEvalKit Evaluation Module (Short Version)

This module evaluates video generation models’ reasoning and provides tools to analyze and visualize results.

## Table of Contents

* [Available Evaluators](#available-evaluators)
* [Analysis Tools](#analysis-tools)
* [Evaluation Criteria](#evaluation-criteria)
* [Output Structure](#output-structure)
* [Command-Line Interface](#command-line-interface)
* [Custom Evaluators](#custom-evaluators)
* [API Reference](#api-reference)
* [Workflow & Best Practices](#workflow--best-practices)
* [Troubleshooting](#troubleshooting)

---

## Available Evaluators

### 1. Human Evaluator

Interactive Gradio UI for human annotation.

**Key Features**

* Side-by-side input / generated video
* Structured 1–5 scoring with comments
* Live progress and per-model stats
* **Automatic resume**: skips tasks with existing eval JSONs
* Annotator name entered directly in the UI

**Usage**

```bash
# Full pilot experiment
python examples/run_evaluation.py human

# Runner module
python -m vmevalkit.runner.evaluate human \
  --experiment pilot_experiment \
  --annotator "John Doe" \
  --port 7860 --share
```

**Python**

```python
from vmevalkit.eval import HumanEvaluator

evaluator = HumanEvaluator(experiment_name="pilot_experiment")
evaluator.launch_interface(share=True, port=7860)
```

---

### 2. GPT-4O Evaluator

Automatic vision-based evaluation using OpenAI GPT-4O.

**Key Features**

* Compares final frame to ground truth
* Domain-specific prompts (chess, maze, rotation, Raven, Sudoku)
* Async batch processing across models
* **Resume support**: saves after each task; skips existing results

**Usage**

```bash
python examples/run_evaluation.py gpt4o
python -m vmevalkit.runner.evaluate gpt4o \
  --experiment pilot_experiment \
  --output-dir data/evaluations \
  --temperature 0.1
```

**Requirements**

* `OPENAI_API_KEY` set
* `opencv-python`, `httpx` installed

---

## Analysis Tools

### `analysis/plot.py` – Performance Visualization

Generates publication-ready plots:

* Overall model success-rate bar chart
* Model × domain heatmap
* Score distributions
* CSV with detailed metrics

```bash
python analysis/plot.py --eval-folder data/evaluations/human-eval/
python analysis/plot.py --eval-folder data/evaluations/gpt4o-eval/
```

### `analysis/stats.py` – Human vs GPT-4O Comparison

Runs statistical comparison between human and GPT-4O scores:

* Basic statistics
* Paired t-test & Wilcoxon
* Pearson / Spearman / Kendall correlations
* Cohen’s kappa
* Bootstrap CIs and convergence analysis
* Scatter plots for papers

```bash
python analysis/stats.py
```

---

## Evaluation Criteria

### 1–5 Solution Correctness Score

* **1**: Completely wrong
* **2**: Mostly wrong, minor correct pieces
* **3**: Partially correct (~half)
* **4**: Mostly correct, minor errors ✓
* **5**: Perfect solution ✓

### Binary Grading (Used in Plots & Stats)

* **Success**: scores **4–5**
* **Failure**: scores **1–3**

---

## Output Structure

### Directory Layout

```text
data/evaluations/
├── human-eval/
│   └── pilot_experiment/...
├── gpt4o-eval/
│   └── pilot_experiment/
│       ├── GPT4OEvaluator_all_models.json
│       └── [same per-task structure]
└── custom-eval/
```

### JSON Schema (Per Task)

```json
{
  "metadata": {
    "evaluator": "human-eval",
    "annotator": "John Doe",
    "timestamp": "2024-10-14T12:00:00",
    "model_name": "luma-ray-2",
    "task_type": "chess_task",
    "task_id": "chess_0000"
  },
  "result": {
    "solution_correctness_score": 5,
    "explanation": "Perfect solution",
    "comments": "User comments",
    "evaluation_type": "final_frame_comparison",
    "status": "completed"
  }
}
```

---

## Command-Line Interface

Unified runner:

```bash
# Human
python -m vmevalkit.runner.evaluate human \
  --experiment pilot_experiment \
  --annotator "John Doe" \
  --port 7860 --share

# GPT-4O
python -m vmevalkit.runner.evaluate gpt4o \
  --experiment pilot_experiment \
  --output-dir data/evaluations \
  --temperature 0.1
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export VMEVAL_DATA_DIR="/path/to/data"         # optional
export VMEVAL_OUTPUT_DIR="/path/to/evaluations" # optional
```

---

## Custom Evaluators

Implement a standalone class that:

1. Uses the same directory structure
2. Writes JSON in the standard format
3. Implements resume via `_has_evaluation()`

Minimal template:

```python
from pathlib import Path
from datetime import datetime
import json

class MyEvaluator:
    def __init__(self, output_dir="data/evaluations/my-eval",
                 experiment_name="pilot_experiment"):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = Path("data/outputs") / experiment_name

    def _has_evaluation(self, model_name, task_type, task_id):
        eval_file = (self.output_dir / self.experiment_name /
                     model_name / task_type / task_id / "MyEvaluator.json")
        return eval_file.exists()

    def evaluate_single(self, model_name, task_type, task_id, video_path):
        return {"solution_correctness_score": 5,
                "explanation": "Perfect", "status": "completed"}

    def _save_result(self, model_name, task_type, task_id, eval_result):
        path = (self.output_dir / self.experiment_name /
                model_name / task_type / task_id)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "MyEvaluator.json", "w") as f:
            json.dump({
                "metadata": {
                    "evaluator": "MyEvaluator",
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "task_type": task_type,
                    "task_id": task_id
                },
                "result": eval_result
            }, f, indent=2)
```

---

## API Reference (High-Level)

* **HumanEvaluator**

  * Params: `output_dir`, `experiment_name`
  * Methods: `launch_interface`, `_load_evaluation_queue`, `_save_evaluation`

* **GPT4OEvaluator**

  * Params: `output_dir`, `experiment_name`, `api_key`, `model`, `temperature`
  * Methods: `extract_final_frame`, `create_prompt`, `evaluate_all_models`,
    `_has_evaluation`, `_save_single_result`

* **EvaluationComparator** (`analysis/stats.py`)

  * Methods: `load_evaluations`, `prepare_paired_data`, `basic_statistics`,
    `paired_t_test`, `correlation_analysis`, `inter_rater_reliability`,
    `bootstrap_confidence_intervals`, `convergence_analysis`, `plot_comparisons`

---

## Workflow & Best Practices

### Recommended Pipeline

```bash
# 1. Generate videos
python examples/experiment_2025-10-14.py

# 2. Human evaluation
python examples/run_evaluation.py human

# 3. GPT-4O evaluation
python examples/run_evaluation.py gpt4o

# 4. Visualizations
python analysis/plot.py --eval-folder data/evaluations/human-eval/
python analysis/plot.py --eval-folder data/evaluations/gpt4o-eval/

# 5. Stats comparison
python analysis/stats.py
```

### Best Practices

* Treat scores **4–5 as success** in all analysis
* Keep experiment names descriptive and consistent
* Use low temperature (≈0.1) for GPT-4O
* Use same annotator for related human tasks when possible
* Commit evaluation JSONs, CSVs, and plots for reproducibility

---

## Troubleshooting

* **Human UI not loading**: check port, install `gradio`, try `--share`.
* **GPT-4O API failures**: verify `OPENAI_API_KEY`, quota, and `opencv-python` / `httpx`.
* **Missing stats**: ensure both human and GPT-4O evaluations exist for the same tasks.
