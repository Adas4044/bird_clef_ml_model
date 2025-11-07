#!/bin/bash
#optimization for M2 Ultra Mac

echo "BirdClef Model Training"

export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24
export VECLIB_MAXIMUM_THREADS=24
export NUMEXPR_NUM_THREADS=24

cd "$(dirname "$0")/src"

python3 train.py \
    --data-dir ../data/birdclef-2025 \
    --output-dir ../outputs \
    --use-sampling \
    --include-all-species \
    --max-samples 500 \
    --target-sr 32000 \
    --target-duration 15.0 \
    --n-mfcc 40 \
    --feature-statistics extended \
    --test-size 0.2 \
    --random-state 42 \
    --n-jobs -1


echo "complete"
