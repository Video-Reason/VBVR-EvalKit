#!/bin/bash
# Generate ALL of Tin's tasks (no limit)
# This will generate the maximum possible samples for each task

set -e  # Exit on error

cd /home/hokindeng/VMEvalKit

echo "=============================================="
echo "Generating Tin's Tasks - ALL SAMPLES (no limit)"
echo "=============================================="
echo ""

echo "1/5 Counting Circles (generating all)..."
./venv/bin/python examples/create_questions.py \
    --task counting_circles

echo ""
echo "2/5 Counting Pentagons (generating all)..."
./venv/bin/python examples/create_questions.py \
    --task counting_pentagons

echo ""
echo "3/5 Counting Squares (generating all)..."
./venv/bin/python examples/create_questions.py \
    --task counting_squares

echo ""
echo "4/5 Letter Counting (generating all)..."
./venv/bin/python examples/create_questions.py \
    --task letter_counting

echo ""
echo "5/5 Subway Pathfinding (generating all)..."
./venv/bin/python examples/create_questions.py \
    --task subway_pathfinding

echo ""
echo "=============================================="
echo "✓ Complete! Generated maximum samples"
echo "=============================================="

