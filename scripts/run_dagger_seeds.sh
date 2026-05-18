#!/bin/bash
# Run DAgger across 3 seeds × 2 init datasets = 6 runs
# Total time: ~3.5 hours on RTX 3050 Ti

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Seed 42 already done — skip
echo "Skipping seed 42 (already complete)"

for SEED in 1 7; do
    for INIT in 100 800; do
        if [ "$INIT" == "800" ]; then
            INITIAL_DATASET="data/demonstrations/demos_train.hdf5"
        else
            INITIAL_DATASET="data/demonstrations/demos_train_${INIT}.hdf5"
        fi
        RUN_NAME="dagger_init${INIT}_seed${SEED}"
        echo ""
        echo "============================================================"
        echo "Run: $RUN_NAME"
        echo "Initial dataset: $INITIAL_DATASET"
        echo "============================================================"
        python scripts/train_dagger.py \
            --config configs/dagger.yaml \
            --initial-dataset "$INITIAL_DATASET" \
            --run-name "$RUN_NAME" \
            --seed "$SEED"
    done
done

echo ""
echo "All DAgger runs complete."