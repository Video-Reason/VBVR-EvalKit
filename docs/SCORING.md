# VMEvalKit Scoring

Comprehensive scoring methods for assessing video generation models' reasoning capabilities.

## Available Evaluators

### Human Evaluation
Interactive Gradio interface for human scoring.

```bash
python examples/score_videos.py --eval-config eval_config.json
# Set "method": "human" in config
```

### GPT-4O Evaluation
Automated scoring using OpenAI's GPT-4O vision model.

```bash
# Requires OPENAI_API_KEY
python examples/score_videos.py --eval-config eval_config.json  
# Set "method": "gpt4o" in config
```

### InternVL Evaluation
Open-source VLM evaluation (requires 30GB VRAM).

```bash
# Start InternVL server
bash script/lmdeploy_server.sh

# Run evaluation
python examples/score_videos.py --eval-config eval_config.json
# Set "method": "internvl" in config
```

### Multi-Frame Evaluation
Advanced evaluation using multiple video frames with consistency analysis and voting.

```bash
# Multi-frame GPT-4O or InternVL
# Set "method": "multiframe_gpt4o" or "multiframe_internvl" in config
```



## Scoring Scale

**1-5 Scale** converted to **Binary** for analysis:
- **Success**: Scores 4-5 (mostly/completely correct)
- **Failure**: Scores 1-3 (incorrect/partially correct)

## Configuration

Create `eval_config.json` to configure evaluation:

```json
{
  "method": "gpt4o",
  "inference_dir": "./outputs", 
  "eval_output_dir": "./evaluations",
  "temperature": 0.0,
  "multiframe": {
    "n_frames": 5,
    "strategy": "hybrid",
    "voting": "weighted_majority"
  }
}
```

## Usage

```bash
# Run evaluation
python examples/score_videos.py --eval-config eval_config.json

# Test multi-frame pipeline (no API calls)
python examples/score_videos.py --test-multiframe --video path/to/video.mp4
```

## Output

Evaluations are saved to the directory specified in `eval_output_dir` (from your config) with structured JSON files containing scores, metadata, and explanations. Results support resume capability and statistical analysis.
