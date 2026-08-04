# UGLY: Uncertainty-Guided Latent Yielding

A small PyTorch prototype for the architecture we discussed:

> System A predicts the answer. System B learns reliability geometry around that prediction.

This repo trains on a controlled 2D two-moons task with synthetic OOD samples so you can quickly inspect whether the uncertainty primitives behave correctly.

## Architecture

```text
x
↓
Encoder / JEPA-style latent backbone
↓
h = latent representation
↓
System A: base prediction path
↓
base logits

System B: uncertainty geometry path
h → K learned internal views
  → M primitive: margin / decision direction
  → D primitive: disagreement between views
  → G primitive: latent geometry spread / instability
  → S primitive: support / knownness
  → primitive attention
  → uncertainty context u
  → gated fusion back into h
↓
final logits
↓
loss
↓
backprop trains both System A and System B
```

The important thing: System B is **inside the forward pass**, not a post-hoc confidence head. Prediction loss can therefore learn whether and when uncertainty geometry helps prediction.

## What the primitives mean

- `M` = margin: how strongly the model leans toward one class.
- `D` = disagreement: whether internal views disagree.
- `G` = geometry/spread: whether latent views are coherent or scattered.
- `S` = support: whether the latent point is in a known/supported region.

The model turns these into primitive tokens, lets them attend to each other, and fuses the resulting uncertainty context back into prediction.

## Why two-moons first?

It is easy to inspect:

- inside the moons: support should be high, uncertainty low
- near the boundary: disagreement/uncertainty should rise
- far from training data: support should be low, uncertainty high
- confident-wrong regions should shrink

## Install

```bash
conda create -n ugly python=3.11 -y
conda activate ugly
pip install -r requirements.txt
```

PyTorch install depends on your CUDA version. If needed, install PyTorch from https://pytorch.org first, then run the requirements file.

## Train UGLY

```bash
python -m src.train --model ugly --epochs 200 --device cuda --out runs/ugly
```

To train the M/D/G primitives with witness-based binary targets instead of the older soft perturbation targets:

```bash
python -m src.train --model ugly --epochs 200 --device cuda --out runs/ugly --use-witness-mdg-loss
```

To run the explicit M-boundary curriculum ablation:

```bash
python -m src.train --model ugly --epochs 200 --device cuda --out runs/ugly_boundary --disable-fusion --disable-primitive-attn --use-boundary-m-loss
```

This keeps support loss active and adds a binary boundary target for `m_scalar`:

- near the current decision boundary -> `m_scalar` should go low
- far from the boundary -> `m_scalar` should go high

## UGLY-Evidence v1

```bash
python -m src.train --model ugly --epochs 3 --device cuda --out runs/smoke_evidence --noise 0.40 --n-train 300 --seed 1 --evidence-mode --use-boundary-m-loss
```

In evidence mode, prediction, uncertainty, support, commitment, and boundary all come from the same shared evidence state:

- `order` drives the class side
- `boundary` carries ambiguity mass
- `support` carries knownness
- final logits are derived from evidence, not from the old final head

When evidence mode is on, the plot outputs also include `order_surface.png` and `boundary_surface.png`.

## Train baseline

```bash
python -m src.train --model baseline --epochs 200 --device cuda --out runs/baseline
```

## Fast smoke test

```bash
python -m src.train --model ugly --epochs 3 --device cpu --out runs/smoke --num-threads 1
```

## Outputs

The training script writes:

```text
runs/<name>/metrics.json
runs/<name>/decision_surface.png
runs/<name>/uncertainty_surface.png
runs/<name>/support_surface.png
runs/<name>/commitment_surface.png
runs/<name>/training_curves.png
runs/<name>/model.pt
```

## What success should look like

UGLY does **not** need to beat the baseline on raw accuracy immediately. The first success is:

- competitive accuracy
- lower confident-wrong rate
- higher uncertainty far from the moons
- higher uncertainty near class boundaries
- lower support on synthetic OOD points
- better risk-coverage curve

## Minimal research claim tested here

> Explicit uncertainty primitives can become useful predictive structure when they are represented as interacting latent tokens and fused into the forward pass.
