# VMEvalKit Evaluation Module

This module provides evaluation methods for assessing video generation models' reasoning capabilities.

## Available Evaluators

### 1. Human Evaluator
Interactive web interface for human annotation of generated videos.

**Features:**
- Gradio-based web interface
- Side-by-side display of input and generated video
- Structured evaluation criteria
- Progress tracking
- Export results as JSON
- **Automatic skip of already evaluated tasks** (checks evaluations folder)
- **In-interface annotator name input** (no command-line setup needed)

**Usage:**
```bash
# Run human evaluation interface (evaluates entire pilot experiment)
# Automatically skips tasks that have any existing evaluation
python examples/run_evaluation.py human
```

When launched, the interface will:
- Prompt for the annotator name
- Automatically skip tasks with existing evaluations
- Show progress and statistics

**Python Usage:**
```python
from vmevalkit.eval import HumanEvaluator

evaluator = HumanEvaluator(
    experiment_name="pilot_experiment"
)
# Automatically skips tasks with existing evaluations in data/evaluations/
# Annotator name is set within the Gradio interface

evaluator.launch_interface(share=True, port=7860)
```

### 2. GPT-4O Evaluator
Automatic evaluation using OpenAI's GPT-4O vision model.

**Features:**
- Compares final frame of generated video with ground truth
- Direct assessment of whether the model answered the question correctly
- Task-specific evaluation prompts
- Batch processing of all models
- Detailed scoring and explanations

**Usage:**
```bash
# Evaluate entire pilot experiment (all models, all tasks)
python examples/run_evaluation.py gpt4o
```

**Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Install dependencies: `opencv-python`, `httpx`

## Evaluation Criteria

Both evaluators use a 1-5 scale for solution correctness:

**Solution Correctness Score**: 
- **1**: Completely wrong solution
- **2**: Mostly incorrect with minor correct elements
- **3**: Partially correct (about half correct)
- **4**: Mostly correct with minor errors
- **5**: Perfect solution

## Output Structure

Evaluations are saved in `data/evaluations/` following this structure:
```
data/evaluations/
├── pilot_experiment/
│   ├── luma-ray-2/
│   │   ├── chess_task/
│   │   │   ├── chess_0000/
│   │   │   │   ├── human-eval.json
│   │   │   │   └── GPT4OEvaluator.json
│   │   │   └── ...
│   │   └── ...
│   └── ...
```

Each `*-eval.json` file contains individual evaluation results with metadata. Summary statistics and analysis should be computed separately from these raw evaluation files.

## Custom Evaluators

To create a custom evaluator, follow the pattern of the existing evaluators:

```python
from pathlib import Path
from datetime import datetime
import json

class MyEvaluator:
    def __init__(self, output_dir="data/evaluations", 
                 experiment_name="pilot_experiment"):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = Path("data/outputs") / experiment_name
    
    def evaluate_single(self, model_name, task_type, task_id, video_path):
        # Your evaluation logic here
        return {
            "solution_correctness_score": 5,  # 1-5 scale
            "explanation": "The solution perfectly solves the task"
        }
    
    def evaluate_model(self, model_name):
        # Iterate through tasks and evaluate
        # Save results using same structure as other evaluators
        pass
```

Note: Evaluators are now standalone classes - no base class required. Just ensure your evaluator saves results in the standard format.

## API Reference

### HumanEvaluator
Gradio-based interface for human annotation.

**Parameters:**
- `output_dir`: Directory for saving evaluations (default: "data/evaluations")
- `experiment_name`: Name of experiment to evaluate (default: "pilot_experiment")

**Methods:**
- `launch_interface(share, port)`: Start web interface with annotator name input

**Automatic Features:**
- Checks `data/evaluations/` for existing `*-eval.json` files
- Automatically skips tasks that have been evaluated
- Logs number of tasks skipped and loaded
- Annotator name is entered via the interface (not constructor)

### GPT4OEvaluator
Automatic evaluation using GPT-4O by comparing final frames.

**Parameters:**
- `output_dir`: Directory for saving evaluations
- `experiment_name`: Name of experiment to evaluate
- `api_key`: OpenAI API key (defaults to OPENAI_API_KEY env var)
- `model`: GPT model to use (default: "gpt-4o")
- `temperature`: Temperature for responses (default: 0.1)

**Methods:**
- `extract_final_frame(video_path)`: Extract the final frame from video
- `create_prompt(task_type)`: Generate task-specific prompts
- `evaluate_single()`: Evaluate one video by comparing final frames
- `evaluate_model()`: Evaluate all tasks for a model
- `evaluate_all_models()`: Evaluate all models in experiment

## Resume Capability

**Human Evaluation:**
- Automatically skips already evaluated tasks
- Checks `data/evaluations/` for any `*-eval.json` files
- Logs how many tasks were skipped
- To re-evaluate: manually delete the evaluation files

**GPT-4O Evaluation:**
- Currently overwrites existing results
- To skip already evaluated: manually check for existing files

## Tips

1. **For Human Evaluation:**
   - The interface shows progress and statistics
   - Complete evaluations in one session when possible
   - Add detailed comments for edge cases
   - Check logs for detailed skip information

2. **For GPT-4O Evaluation:**
   - Monitor API costs for large experiments
   - Results are deterministic with low temperature (0.1)
   - Only tasks with ground truth final frames will be evaluated
   - Focuses on final result rather than process

3. **General:**
   - Run evaluations after all inference is complete
   - Compare human and GPT-4O results for validation
   - Usage is simple: `python examples/run_evaluation.py <method>`