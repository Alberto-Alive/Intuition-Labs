# RFQ — Recursive Functional Quotienting

RFQ is a research framework for discovering alternative functional subnetworks
inside an ordinary trained neural network, quotienting genuinely redundant
strategies, retaining a minimal complementary strategy basis, and routing each
input through an appropriate surviving strategy.

The core pipeline is:

    dense model
        -> population perturbations
        -> strategy discovery
        -> functional equivalence quotienting
        -> complementary basis selection
        -> sparse conditional routing

This package contains:

- a reusable Python implementation of the RFQ stages,
- a strictly split, end-to-end digits experiment for fast validation,
- a CIFAR-10 / ResNet-18 publication experiment scaffold,
- unit tests,
- a publication protocol and ablation plan.

## What is already implemented

1. **Population Strategy Discovery**
   - Sample structural masks.
   - Evaluate them on a discovery split.
   - Cluster successful masks.
   - Greedily minimize each family into a sparse working core.

2. **Functional Equivalence Quotienting**
   - Compare cores by behavior across a transformation suite.
   - Group highly equivalent strategies.
   - Keep the best representative of each equivalence class.

3. **Complementary Basis Selection**
   - Search subsets of quotient representatives.
   - Prefer the smallest subset that preserves a target level of behavior.
   - Report clean accuracy, transformed accuracy, fidelity, oracle coverage,
     storage proxy, and Pareto frontiers.

4. **Sparse Conditional Routing**
   - Train a cheap router on a separate router-training split.
   - Route each input to one retained strategy.

## Fast validation

Run:

```bash
pip install -r requirements.txt
python scripts/run_digits_clean_pipeline.py --output-dir results/digits_clean
pytest -q
```

The digits pipeline uses strict split separation:

- dense-model training split,
- strategy-discovery split,
- quotient/basis split,
- router-training split,
- final test split touched only at the end.

## Publication-scale experiment

The first serious benchmark is configured around CIFAR-10 and a gated ResNet-18.

See:

- `docs/PUBLICATION_PROTOCOL.md`
- `docs/THEORY.md`
- `configs/cifar10_resnet18.json`

The CIFAR runner is intentionally written as an experimental scaffold rather
than pretending that the publication-scale run has already been executed here.

## Scientific claim to test

The strongest intended claim is not "another pruning method."

It is:

> Neural networks can contain structurally different internal subnetworks that
> are functionally redundant or complementary. These alternatives can be
> discovered by intervention, quotiented by functional equivalence, compressed
> recursively, and selectively routed at inference time.

RFQ must beat strong baselines under a clean held-out protocol before that claim
is treated as established.
