from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .basis import BasisConfig, select_complementary_basis
from .discovery import DiscoveryConfig, discover_strategies
from .quotient import QuotientConfig, quotient_strategies
from .router import evaluate_router, train_router
from .utils import save_json, seed_everything


torch.set_num_threads(1)


class MaskableDigitsMLP(nn.Module):
    def __init__(self, h1: int = 48, h2: int = 24):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.fc1 = nn.Linear(64, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 10)

    @property
    def n_modules(self) -> int:
        return self.h1 + self.h2

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h1 = F.relu(self.fc1(x))
        if mask is not None:
            h1 = h1 * mask[: self.h1]
        h2 = F.relu(self.fc2(h1))
        if mask is not None:
            h2 = h2 * mask[self.h1 :]
        return self.fc3(h2)


def split_data(seed: int = 0):
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data).astype(np.float32)
    y = digits.target.astype(np.int64)

    # Final test is split first and never used for discovery/quotient/router.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    # Remaining data is partitioned into four disjoint development roles.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X_dev,
        y_dev,
        test_size=0.42,
        random_state=seed + 1,
        stratify=y_dev,
    )

    X_discovery, X_rest2, y_discovery, y_rest2 = train_test_split(
        X_rest,
        y_rest,
        test_size=0.50,
        random_state=seed + 2,
        stratify=y_rest,
    )

    X_quotient, X_router, y_quotient, y_router = train_test_split(
        X_rest2,
        y_rest2,
        test_size=0.50,
        random_state=seed + 3,
        stratify=y_rest2,
    )

    def t(x, y):
        return torch.tensor(x), torch.tensor(y)

    return {
        "train": t(X_train, y_train),
        "discovery": t(X_discovery, y_discovery),
        "quotient": t(X_quotient, y_quotient),
        "router": t(X_router, y_router),
        "test": t(X_test, y_test),
    }


def train_dense(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 350,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=0.015)
    for _ in range(epochs):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def accuracy(model, X, y, mask=None) -> float:
    with torch.no_grad():
        logits = model(X, mask)
        return float((logits.argmax(1) == y).float().mean().item())


def eval_masks(
    model,
    X,
    y,
    masks: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Vectorized masked evaluation for the two-hidden-layer digits MLP.
    """
    masks = np.asarray(masks, dtype=np.float32)
    values = []

    with torch.no_grad():
        h1_base = F.relu(model.fc1(X))

        for start in range(0, len(masks), batch_size):
            m = torch.tensor(
                masks[start : start + batch_size],
                dtype=torch.float32,
            )
            m1 = m[:, : model.h1]
            m2 = m[:, model.h1 :]

            h1 = h1_base.unsqueeze(0) * m1.unsqueeze(1)
            h2 = F.relu(
                torch.einsum(
                    "bnh,kh->bnk",
                    h1,
                    model.fc2.weight,
                )
                + model.fc2.bias
            )
            h2 = h2 * m2.unsqueeze(1)

            logits = (
                torch.einsum(
                    "bnk,ck->bnc",
                    h2,
                    model.fc3.weight,
                )
                + model.fc3.bias
            )

            pred = logits.argmax(dim=-1)
            acc = (
                pred == y.unsqueeze(0)
            ).float().mean(dim=1)

            values.append(acc.cpu().numpy())

    return np.concatenate(values)


def transform_suite(X: torch.Tensor, seed: int):
    rng = np.random.default_rng(seed)

    def shift(x, dx=0, dy=0):
        imgs = x.reshape(-1, 8, 8)
        out = torch.zeros_like(imgs)
        xs_src = slice(max(0, -dx), min(8, 8 - dx))
        xs_dst = slice(max(0, dx), min(8, 8 + dx))
        ys_src = slice(max(0, -dy), min(8, 8 - dy))
        ys_dst = slice(max(0, dy), min(8, 8 + dy))
        out[:, ys_dst, xs_dst] = imgs[:, ys_src, xs_src]
        return out.reshape(-1, 64)

    suite = {"clean": X}

    for sigma in (0.15, 0.30):
        noise = torch.tensor(
            rng.normal(0, sigma, X.shape),
            dtype=torch.float32,
        )
        suite[f"gaussian_{sigma}"] = X + noise

    for drop in (0.10, 0.25):
        keep = torch.tensor(
            rng.random(X.shape) > drop,
            dtype=torch.float32,
        )
        suite[f"dropout_{drop}"] = X * keep

    suite["shift_left"] = shift(X, dx=-1)
    suite["shift_right"] = shift(X, dx=1)
    suite["shift_up"] = shift(X, dy=-1)
    suite["shift_down"] = shift(X, dy=1)

    return suite


def behavior_record(model, mask, suite, y):
    behaviors = {}
    with torch.no_grad():
        for condition, X in suite.items():
            logits = model(X, mask)
            probs = torch.softmax(logits, dim=1)
            behaviors[condition] = {
                "pred": logits.argmax(1).cpu().numpy(),
                "probs": probs.cpu().numpy(),
                "logits": logits.cpu().numpy(),
            }
    return behaviors


def run_digits_pipeline(
    output_dir: str | Path,
    seed: int = 0,
    n_masks: int = 5000,
    n_clusters: int = 5,
    dense_epochs: int = 350,
    router_epochs: int = 300,
    max_minimization_steps: int | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    splits = split_data(seed)

    model = MaskableDigitsMLP()
    train_dense(model, *splits["train"], epochs=dense_epochs)
    model.eval()

    discovery_X, discovery_y = splits["discovery"]
    full_discovery = accuracy(model, discovery_X, discovery_y)

    discovery_config = DiscoveryConfig(
        n_masks=n_masks,
        n_clusters=n_clusters,
        success_tolerance=0.05,
        random_state=seed,
        max_minimization_steps=max_minimization_steps,
    )

    rng = np.random.default_rng(seed + 100)

    strategies, discovery_meta = discover_strategies(
        n_modules=model.n_modules,
        full_score=full_discovery,
        evaluate_many=lambda masks: eval_masks(
            model,
            discovery_X,
            discovery_y,
            masks,
        ),
        evaluate_one=lambda mask: accuracy(
            model,
            discovery_X,
            discovery_y,
            torch.tensor(mask, dtype=torch.float32),
        ),
        config=discovery_config,
        rng=rng,
    )

    quotient_X, quotient_y = splits["quotient"]
    quotient_suite = transform_suite(quotient_X, seed + 200)

    records = []

    for strategy in strategies:
        mask_t = torch.tensor(strategy.mask, dtype=torch.float32)
        records.append({
            "mask": strategy.mask,
            "score": strategy.score,
            "cost": int(strategy.mask.sum()),
            "behaviors": behavior_record(
                model,
                mask_t,
                quotient_suite,
                quotient_y,
            ),
        })

    quotient = quotient_strategies(
        records,
        QuotientConfig(
            min_prediction_agreement=0.965,
            max_js_divergence=0.04,
        ),
    )

    rep_indices = quotient["representatives"]

    full_quotient_logits = {}
    with torch.no_grad():
        for condition, X in quotient_suite.items():
            full_quotient_logits[condition] = model(X).cpu().numpy()

    strategy_logits = []
    strategy_costs = []

    for idx in rep_indices:
        strategy_logits.append({
            condition: records[idx]["behaviors"][condition]["logits"]
            for condition in quotient_suite
        })
        strategy_costs.append(float(records[idx]["cost"]))

    basis = select_complementary_basis(
        strategy_logits=strategy_logits,
        full_logits=full_quotient_logits,
        labels=quotient_y.cpu().numpy(),
        costs=strategy_costs,
        config=BasisConfig(target_tolerance=0.02),
    )

    chosen_record = (
        basis["smallest_near_full"]
        or basis["best_accuracy"]
    )

    chosen_rep_positions = chosen_record["strategies"]
    chosen_original_indices = [
        rep_indices[pos] for pos in chosen_rep_positions
    ]

    chosen_masks = [
        torch.tensor(
            strategies[idx].mask,
            dtype=torch.float32,
        )
        for idx in chosen_original_indices
    ]

    # Router train split only.
    router_X, router_y = splits["router"]
    router_suite = transform_suite(router_X, seed + 300)

    router_features = []
    router_logits = []
    router_labels = []

    with torch.no_grad():
        for condition, X in router_suite.items():
            stack = torch.stack(
                [model(X, mask) for mask in chosen_masks],
                dim=1,
            )
            router_features.append(X)
            router_logits.append(stack)
            router_labels.append(router_y)

    router_features = torch.cat(router_features, dim=0)
    router_logits = torch.cat(router_logits, dim=0)
    router_labels = torch.cat(router_labels, dim=0)

    router = train_router(
        features=router_features,
        strategy_logits=router_logits,
        labels=router_labels,
        hidden_dim=24,
        epochs=router_epochs,
    )

    # Final test is touched only here.
    test_X, test_y = splits["test"]
    test_suite = transform_suite(test_X, seed + 400)

    final_conditions = {}

    with torch.no_grad():
        for condition, X in test_suite.items():
            full_logits = model(X)
            stack = torch.stack(
                [model(X, mask) for mask in chosen_masks],
                dim=1,
            )
            routed = evaluate_router(
                router=router,
                features=X,
                strategy_logits=stack,
                labels=test_y,
            )

            final_conditions[condition] = {
                "full_accuracy": float(
                    (full_logits.argmax(1) == test_y)
                    .float()
                    .mean()
                    .item()
                ),
                **routed,
            }

    result = {
        "seed": seed,
        "split_sizes": {
            name: len(data[0])
            for name, data in splits.items()
        },
        "discovery_config": asdict(discovery_config),
        "discovery": discovery_meta,
        "strategies": [
            {
                "cluster": s.cluster,
                "score": s.score,
                "modules": np.flatnonzero(s.mask).tolist(),
                "size": int(s.mask.sum()),
            }
            for s in strategies
        ],
        "quotient": {
            "equivalence_classes": quotient["equivalence_classes"],
            "representatives": rep_indices,
            "pairwise": quotient["pairwise"],
        },
        "basis": basis,
        "chosen_original_strategy_indices": chosen_original_indices,
        "final_test": {
            "per_condition": final_conditions,
            "full_mean_accuracy": float(
                np.mean([
                    v["full_accuracy"]
                    for v in final_conditions.values()
                ])
            ),
            "routed_mean_accuracy": float(
                np.mean([
                    v["routed_accuracy"]
                    for v in final_conditions.values()
                ])
            ),
            "ensemble_mean_accuracy": float(
                np.mean([
                    v["ensemble_accuracy"]
                    for v in final_conditions.values()
                ])
            ),
            "oracle_mean_coverage": float(
                np.mean([
                    v["oracle_coverage"]
                    for v in final_conditions.values()
                ])
            ),
        },
    }

    save_json(output_dir / "digits_clean_pipeline_result.json", result)
    torch.save(
        {
            "model": model.state_dict(),
            "router": router.state_dict(),
            "chosen_masks": chosen_masks,
        },
        output_dir / "digits_clean_pipeline_checkpoint.pt",
    )

    return result
