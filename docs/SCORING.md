# VMEvalKit Scoring Module

This module provides comprehensive scoring methods for assessing video generation models' reasoning capabilities, along with powerful analysis tools for processing and visualizing results.

## Table of Contents
- [Available Scorers](#available-scorers)
- [Analysis Tools](#analysis-tools)
- [Scoring Criteria](#scoring-criteria)
- [Output Structure](#output-structure)
- [Command-Line Interface](#command-line-interface)
- [Custom Scorers](#custom-scorers)
- [API Reference](#api-reference)
- [Resume Capability](#resume-capability)
- [Analysis Workflow](#analysis-workflow)
- [Tips and Best Practices](#tips-and-best-practices)

## Available Scorers

### 1. Human Scorer
Interactive web interface for human scoring of generated videos.

**Usage:**
```bash
# Run human scoring interface (scores entire pilot experiment)
# Automatically skips tasks that have any existing scoring
python examples/score_videos.py human

# Or use the command-line runner
python -m vmevalkit.runner.score human --annotator "John Doe" --port 7860 --share
```

When launched, the interface will:
- Prompt for the annotator name
- Automatically skip tasks with existing scorings
- Show progress and statistics by model
- Display completion percentages

**Python Usage:**
```python
from vmevalkit.eval import HumanScorer

scorer = HumanScorer(
    experiment_name="pilot_experiment"
)
# Automatically skips tasks with existing scorings in data/scorings/
# Annotator name is set within the Gradio interface

scorer.launch_interface(share=True, port=7860)
```

### 2. GPT-4O Scorer
Automatic scoring using OpenAI's GPT-4O vision model.

**Usage:**
```bash
# Evaluate entire pilot experiment (all models, all tasks)
python examples/score_videos.py gpt4o
```

**Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Install dependencies: `opencv-python`, `httpx`

### 3. InternVL3-8B Scorer



###  Opensource Evaluator


refer https://huggingface.co/OpenGVLab/InternVL3-8B

```bash
uv pip install lmdeploy timm peft>=0.17.0 openai
CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server OpenGVLab/InternVL3-8B --chat-template internvl2_5 --server-port 23333 --tp 1 # takes 30GB vram.

# in another terminal
uv run examples/score_videos.py internvl
```

After install the dependencies, run the following commands to run the batch evaluation.
```
cd VMEvalKit

bash script/run.sh

bash script/lmdeploy_server.sh
```



## Analysis Tools

### Performance Visualization (`analysis/plot.py`)
Creates professional visualizations of scoring results.

**Usage:**
```bash
# Analyze human scoring results
python analysis/plot.py --eval-folder data/scorings/human-eval/

# Analyze GPT-4O scoring results
python analysis/plot.py --eval-folder data/scorings/gpt4o-eval/
```

**Outputs:**
1. **Overall Performance Bar Chart**: Shows success rates for all models
2. **Domain Performance Heatmap**: Model × Domain matrix visualization
3. **Score Distribution Plots**: Distribution of scores per model
4. **Detailed Statistics CSV**: Complete performance metrics

### Statistical Comparison Tool (`analysis/stats.py`)
Comprehensive statistical analysis comparing GPT-4O and human scorings.

**Usage:**
```bash
# Run full statistical comparison
python analysis/stats.py
```

**Statistical Tests Performed:**
1. **Basic Statistics**: Mean, std, median for both scorers
2. **Paired T-Test**: Tests if means are significantly different
3. **Wilcoxon Test**: Non-parametric alternative
4. **Correlation Analysis**: Measures agreement strength
5. **Inter-rater Reliability**: Cohen's kappa coefficient
6. **Bootstrap CI**: 95% confidence intervals for mean difference
7. **Convergence Analysis**: Finds sample size for statistical equivalence

## Scoring Criteria

### Scoring Scale (1-5)
Both scorers use a 1-5 scale for solution correctness:

**Solution Correctness Score**: 
- **1**: Completely wrong solution - no understanding of task
- **2**: Mostly incorrect with minor correct elements
- **3**: Partially correct (about half correct)
- **4**: Mostly correct with minor errors ✓
- **5**: Perfect solution ✓

### Binary Grading System
**Important**: For final analysis, scores are converted to binary:
- **Success (Correct)**: Scores 4 and 5
- **Failure (Incorrect)**: Scores 1, 2, and 3

This binary system is used in all performance calculations and visualizations.

## Output Structure

### Directory Structure
Scorings are saved in `data/scorings/` following this structure:
```
data/scorings/
├── human-eval/              # Human scorings
│   └── pilot_experiment/
│       ├── luma-ray-2/
│       │   ├── chess_task/
│       │   │   ├── chess_0000/
│       │   │   │   └── human-eval.json
│       │   │   └── ...
│       │   └── ...
│       └── ...
├── gpt4o-eval/              # GPT-4O scorings  
│   └── pilot_experiment/
│       ├── GPT4OScorer_all_models.json  # Aggregated results
│       └── [same structure as human-eval]
└── custom-eval/             # Custom scorings
```

### JSON File Format
Each scoring JSON file contains:
```json
{
  "metadata": {
    "scorer": "human-eval",        // or "GPT4OScorer"
    "annotator": "John Doe",           // (human only)
    "timestamp": "2024-10-14T12:00:00",
    "model_name": "luma-ray-2",
    "task_type": "chess_task",
    "task_id": "chess_0000"
  },
  "result": {
    "solution_correctness_score": 5,  // 1-5 scale
    "explanation": "Perfect solution", // (GPT-4O only)
    "comments": "User comments",      // (human only)
    "scoring_type": "final_frame_comparison", // (GPT-4O only)
    "status": "completed"              // or "failed", "skipped"
  }
}
```

## Command-Line Interface

### Using the Runner Module
The `vmevalkit.runner.score` module provides a unified CLI:

```bash
# Human scoring with options
python -m vmevalkit.runner.score human \
  --experiment pilot_experiment \
  --annotator "John Doe" \
  --port 7860 \
  --share

# GPT-4O scoring with options
python -m vmevalkit.runner.score gpt4o \
  --experiment pilot_experiment \
  --output-dir data/scorings \
```

### Environment Variables
```bash
# Required for GPT-4O scoring
export OPENAI_API_KEY="your-api-key"

# Optional: Set default paths
export VMEVAL_DATA_DIR="/path/to/data"
export VMEVAL_OUTPUT_DIR="/path/to/scorings"
```

## Analysis Workflow

### Complete Scoring Pipeline
```bash
# 1. Run inference to generate videos
python examples/experiment_2025-10-14.py

# 2. Run human scoring (interactive)
python examples/score_videos.py human

# 3. Run GPT-4O scoring (automatic)
python examples/score_videos.py gpt4o

# 4. Visualize results
python analysis/plot.py --eval-folder data/scorings/human-eval/
python analysis/plot.py --eval-folder data/scorings/gpt4o-eval/

# 5. Compare human vs GPT-4O statistically
python analysis/stats.py
```

### Output Files Generated
1. **Scoring JSONs**: Individual task scorings
2. **Performance Charts**: PNG and EPS visualizations
3. **Statistical Results**: JSON with comparison metrics
4. **CSV Reports**: Detailed performance tables

## Tips and Best Practices

### For Human Scoring
- **Session Management**: Complete scorings in one session when possible
- **Annotator Consistency**: Use same annotator for related tasks
- **Comments**: Add detailed comments for edge cases or ambiguous results
- **Progress Monitoring**: Use refresh button to update queue status
- **Backup**: Scorings are saved immediately after submission

### For GPT-4O Scoring
- **Cost Management**: Monitor API usage for large experiments
- **Ground Truth**: Only tasks with final frame ground truth will be scored
- **Resume on Failure**: Safe to restart after interruptions

### For Analysis
- **Binary Grading**: Remember scores 4-5 are considered success
- **Statistical Significance**: Use stats.py to validate comparisons
- **Visualization**: Use plot.py for publication-ready figures
- **Domain Analysis**: Check performance variations across task types

### General Best Practices
1. **Order of Operations**: Always run inference → scoring → analysis
2. **Experiment Naming**: Use descriptive experiment names for organization
3. **Version Control**: Track scoring results in git for reproducibility
4. **Documentation**: Document any custom scoring criteria
5. **Validation**: Compare human and GPT-4O results for validation
6. **Error Handling**: Check logs for failed scorings
7. **Data Backup**: Regularly backup scoring results

## Troubleshooting

### Common Issues and Solutions

**Human Scoring Interface Not Loading:**
- Check if port 7860 is available
- Try using `--share` flag for public URL
- Verify Gradio is installed: `pip install gradio`

**GPT-4O API Errors:**
- Verify OPENAI_API_KEY is set correctly
- Check API quota and rate limits
- Ensure opencv-python is installed

**Missing Scorings:**
- Check scoring directory structure
- Verify experiment name matches
- Look for error messages in logs

**Statistical Analysis Failures:**
- Ensure paired scorings exist
- Check for sufficient sample size
- Verify data format consistency