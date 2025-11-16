# MME-CoF Dataset Generation Guide

This guide explains how to generate the MME-CoF dataset with solution images in VMEvalKit format and upload it to HuggingFace.

## Overview

The original MME-CoF dataset contains:
- 59 reasoning puzzle images
- Category labels (12 categories)
- **No solution images** ❌

We transform it into VMEvalKit format:
- 59 task pairs: `first_frame.png` → `final_frame.png`
- Category-specific prompts
- **LLM-generated solution images** ✅

## Prerequisites

### 1. Install Dependencies

```bash
pip install google-generativeai huggingface-hub datasets pillow tqdm
```

### 2. Set Up API Keys

```bash
# Gemini API Key (required for solution generation)
export GEMINI_API_KEY='your-gemini-api-key'

# HuggingFace Token (required for upload)
export HF_TOKEN='your-huggingface-token'
huggingface-cli login
```

Get your API keys:
- **Gemini**: https://aistudio.google.com/app/apikey
- **HuggingFace**: https://huggingface.co/settings/tokens

## Generation Pipeline

### Step 1: Generate Solutions Locally

```bash
cd /Users/access/VMEvalKit

# Option A: Use Imagen 3 for high-quality solution images (recommended, slower)
python vmevalkit/tasks/mme_cof_task/generate_dataset.py \
    --output-dir ./data/mme_cof_generated \
    --use-imagen

# Option B: Use text annotations (faster, cheaper)
python vmevalkit/tasks/mme_cof_task/generate_dataset.py \
    --output-dir ./data/mme_cof_generated
```

**What happens:**
1. Downloads MME-CoF dataset (59 images)
2. For each image:
   - Analyzes puzzle using **Gemini 2.0 Flash**
   - Generates solution description
   - Creates solution image via **Imagen 3** (or annotated overlay)
   - Generates category-specific prompt
   - Saves everything in VMEvalKit format

**Expected time:**
- With Imagen: ~5-10 minutes (with API rate limits)
- Without Imagen: ~2-3 minutes

**Output structure:**
```
data/mme_cof_generated/
├── mme_cof_0000/
│   ├── first_frame.png          # Original puzzle
│   ├── final_frame.png           # Generated solution
│   ├── prompt.txt                # Category-specific prompt
│   └── question_metadata.json   # Metadata with category
├── mme_cof_0001/
│   ├── first_frame.png
│   ├── final_frame.png
│   ├── prompt.txt
│   └── question_metadata.json
├── ...
└── dataset_summary.json         # Statistics and info
```

### Step 2: Review Generated Solutions

```bash
# Check the summary
cat data/mme_cof_generated/dataset_summary.json

# Review some examples
open data/mme_cof_generated/mme_cof_0000/first_frame.png
open data/mme_cof_generated/mme_cof_0000/final_frame.png
```

**Quality check:**
- Do solution images make sense?
- Are prompts appropriate for each category?
- Are all 59 tasks successfully generated?

### Step 3: Upload to HuggingFace

```bash
# Upload to your organization
python vmevalkit/tasks/mme_cof_task/generate_dataset.py \
    --output-dir ./data/mme_cof_generated \
    --upload \
    --hf-dataset-name "YourOrg/MME-CoF-VMEval"
```

**Dataset card will include:**
- Original source attribution
- Generation methodology
- Category breakdown
- Usage examples

## Using the Generated Dataset

Once uploaded, use it in VMEvalKit:

```python
# In vmevalkit/utils/constant.py
'mme_cof': {
    'name': 'MME-CoF',
    'description': 'Video Chain-of-Frame reasoning with generated solutions',
    'hf': True,
    'hf_dataset': 'YourOrg/MME-CoF-VMEval',  # Your dataset!
    'hf_split': 'train',
    'hf_prompt_column': 'prompt',
    'hf_image_column': 'image',
    'hf_solution_image_column': 'solution_image',
}
```

Then download like any other task:

```bash
python examples/create_questions.py --task mme_cof
```

## Solution Generation Strategy

### How Solutions Are Generated

For each puzzle image, we use a **two-stage LLM pipeline**:

**Stage 1: Analysis (Gemini 2.0 Flash)**
```python
Prompt: "Analyze this [category] puzzle and describe 
         the FINAL SOLVED STATE in detail."

Output: Detailed textual description of solution
```

**Stage 2: Image Generation (Imagen 3)**
```python
Input: Original image + solution description
Output: Visual representation of solved state
```

### Category-Specific Prompts

Each of the 12 categories gets specialized prompts:

| Category | Example Prompt |
|----------|---------------|
| 2D Geometry | "Analyze this 2D geometry puzzle. Describe what geometric transformation is required..." |
| Chess | "Analyze this chess position. Identify the best move and describe the final board state..." |
| Visual Trace | "Analyze this path tracing puzzle. Describe the completed path..." |
| ... | ... |

See `PROMPTS.py` for full prompt templates.

## Cost Estimation

**API Costs (approximate):**
- Gemini 2.0 Flash: ~$0.10 per 59 images (analysis)
- Imagen 3: ~$0.20-$0.40 per 59 images (generation)
- **Total: ~$0.30-$0.50** for complete dataset

**Without Imagen (text annotations only): ~$0.10 total**

## Regenerating Specific Categories

To regenerate solutions for specific categories:

```python
from vmevalkit.tasks.mme_cof_task.generate_dataset import generate_solutions_for_dataset
from datasets import load_dataset

dataset = load_dataset("ZiyuG/MME-CoF", split="train")

# Filter to specific category
chess_only = dataset.filter(lambda x: x['label'] == 'chess')

generate_solutions_for_dataset(
    chess_only,
    output_dir="./data/mme_cof_chess_only",
    use_imagen=True
)
```

## Troubleshooting

### Issue: "API Key not found"
```bash
export GEMINI_API_KEY='your-key-here'
# or
export GOOGLE_API_KEY='your-key-here'
```

### Issue: Rate limit errors
The script includes automatic rate limiting (1 second between requests). If you still hit limits:
- Add longer delays in `generate_dataset.py`
- Use `--skip-existing` to resume interrupted runs

### Issue: Imagen not available
If Imagen API is not available in your region:
- Use `--no-imagen` flag for text annotations
- Solutions will have description overlays instead

### Issue: Some solutions look wrong
- Review problematic categories
- Adjust prompts in `PROMPTS.py`
- Regenerate specific tasks manually

## Dataset License

The generated dataset should:
1. ✅ Attribute original MME-CoF dataset
2. ✅ Note that solutions are LLM-generated
3. ✅ Use compatible license (MIT recommended)

Include this in your HuggingFace dataset card:

```markdown
## Dataset Details

- **Original Dataset**: MME-CoF by Guo et al.
- **Source**: https://huggingface.co/datasets/ZiyuG/MME-CoF
- **Modifications**: Added LLM-generated solution images using Gemini 2.0 + Imagen 3
- **Format**: VMEvalKit compatible (image pairs with prompts)
```

## Next Steps

After generation and upload:

1. **Test the dataset**: Download and test in VMEvalKit
2. **Update documentation**: Add dataset to main README
3. **Share with community**: Announce on HuggingFace discussions
4. **Iterate**: Improve prompts based on user feedback

## References

- Original MME-CoF: https://github.com/ZiyuGuo99/MME-CoF
- Paper: "Are Video Models Ready as Zero-Shot Reasoners?"
- Dataset: https://huggingface.co/datasets/ZiyuG/MME-CoF

