#!/bin/bash
# Generate all of Tin's tasks with optimal sample counts
# Total: ~872 samples across 5 tasks

set -e  # Exit on error

cd /home/hokindeng/VMEvalKit

echo "========================================="
echo "Generating Tin's Tasks - Optimal Counts"
echo "========================================="
echo ""

echo "1/5 Counting Circles (160 samples)..."
./venv/bin/python examples/create_questions.py \
    --task counting_circles \
    --pairs-per-domain 160

echo ""
echo "2/5 Counting Pentagons (12 samples)..."
./venv/bin/python examples/create_questions.py \
    --task counting_pentagons \
    --pairs-per-domain 12

echo ""
echo "3/5 Counting Squares (120 samples)..."
./venv/bin/python examples/create_questions.py \
    --task counting_squares \
    --pairs-per-domain 120

echo ""
echo "4/5 Letter Counting (400 samples)..."
./venv/bin/python examples/create_questions.py \
    --task letter_counting \
    --pairs-per-domain 400

echo ""
echo "5/5 Subway Pathfinding (180 samples)..."
./venv/bin/python examples/create_questions.py \
    --task subway_pathfinding \
    --pairs-per-domain 180

echo ""
echo "========================================="
echo "✓ Complete! Generated ~872 samples total"
echo "========================================="
echo ""
echo "Data saved in: data/questions/"
echo "  - counting_circles_task/"
echo "  - counting_pentagons_task/"
echo "  - counting_squares_task/"
echo "  - letter_counting_task/"
echo "  - subway_pathfinding_task/"

