#!/bin/sh
# Experiment B batch sequence (after the seed-0 screen:
#   python run_experiment_b.py --seeds 0 --scales 4,8,16 --tag screen
# which, like the medium batch below, ran with the initial FQC settings
# --fqc-rounds 2 --fqc-jitter 0).
set -e
cd "$(dirname "$0")"

# Medium validation: two more seeds over the full mode matrix (R=2, no jitter).
python run_experiment_b.py --seeds 1,2 --scales 4,8,16 --fqc-rounds 2 --fqc-jitter 0 --tag medium

# Duplicated-evidence stress: index-distinct != content-distinct (R=2).
python run_experiment_b.py --seeds 0,1,2 --scales 8 --duplicated-evidence \
    --modes learned,learned_distinct,fqc,fqc_lexical,fqc_shuffled \
    --fqc-rounds 2 --fqc-jitter 0 --tag stress

# Relevance-warmed routers: does FQC + trained relevance beat forced distinct?
python run_experiment_b.py --seeds 0,1,2 --scales 8,16 --support-warmup-epochs 4 \
    --modes learned,learned_distinct,fqc,fqc_shuffled \
    --fqc-rounds 2 --fqc-jitter 0 --tag warm

# Scale probe at N=32.
python run_experiment_b.py --seeds 0 --scales 32 \
    --modes learned,learned_distinct,fqc --fqc-rounds 2 --fqc-jitter 0 --tag n32

# Corrected FQC (6 quotient rounds + private re-pick jitter, restoring the
# Experiment A symmetry-breaking design at eval time).
python run_experiment_b.py --seeds 0,1,2 --scales 4,8,16 \
    --modes fqc,fqc_shuffled,fqc_random_reassign,fqc_lexical --tag standard_r6
python run_experiment_b.py --seeds 0,1,2 --scales 8 --duplicated-evidence \
    --modes fqc,fqc_lexical,fqc_shuffled --tag stress_r6

# Headline confirmation with a 3x larger test set (192 examples).
python run_experiment_b.py --seeds 0,1,2 --scales 4 --n-test 192 \
    --modes learned,learned_distinct,fqc --tag confirm_r6
python run_experiment_b.py --seeds 0,1,2 --scales 4 --n-test 192 \
    --modes fqc --fqc-rounds 2 --fqc-jitter 0 --tag confirm_r2
