# Installing Dependencies for Tin's Tasks

The Tin's tasks require some additional Python packages that may not be in your venv yet.

## Required Dependencies

```bash
cd /home/hokindeng/VMEvalKit
./venv/bin/pip install matplotlib pillow numpy tqdm
```

## Verify Installation

```bash
./venv/bin/python -c "import matplotlib; import PIL; import numpy; print('✓ All dependencies installed!')"
```

## Generate Datasets

Once dependencies are installed, you have two options:

### Option 1: Generate Optimal Counts (~872 samples)
```bash
./generate_tin_tasks.sh
```

This generates:
- Counting Circles: 160 samples
- Counting Pentagons: 12 samples
- Counting Squares: 120 samples
- Letter Counting: 400 samples
- Subway Pathfinding: 180 samples

### Option 2: Generate ALL Possible Samples (no limit)
```bash
./generate_tin_tasks_all.sh
```

This generates the maximum possible samples for each task.

### Option 3: Generate Individual Tasks
```bash
# Example: Generate just counting circles with 10 samples for testing
./venv/bin/python examples/create_questions.py \
    --task counting_circles \
    --pairs-per-domain 10
```

## Check Generated Data

```bash
ls -la data/questions/counting_circles_task/
ls -la data/questions/counting_pentagons_task/
ls -la data/questions/counting_squares_task/
ls -la data/questions/letter_counting_task/
ls -la data/questions/subway_pathfinding_task/
```

## Troubleshooting

### Missing matplotlib
```bash
./venv/bin/pip install matplotlib
```

### Missing PIL/Pillow
```bash
./venv/bin/pip install Pillow
```

### Missing numpy
```bash
./venv/bin/pip install numpy
```

### Missing tqdm
```bash
./venv/bin/pip install tqdm
```

