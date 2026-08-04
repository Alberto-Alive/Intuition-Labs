"""Membership Inference Attack — harness-compliant implementation.

Metrics (frozen in harness.md v1.0)
------------------------------------
Primary   : AUROC
Secondary : attack_advantage = 2 * (AUROC - 0.5)
            TPR @ 1% FPR
            TPR @ 5% FPR

Attack strategies
-----------------
Two complementary attacks are run; the stronger (higher AUROC) is reported.

1. Direct attack (baseline signal check):
   For each target record, construct a query matching the record's subgroup,
   observe the primitive output, and train LR + MLP on
   [record_features, primitive_outputs].
   Expected to be weak for group query systems with min_group_size ≥ 10.

2. Shadow-dataset attack (gold standard for group query systems):
   Create K shadow private datasets (random N/2 subsets of the combined
   member + non-member pool).  For each shadow dataset D_k, issue M queries
   per target record targeting the record's attribute subgroup.
   Attack features per record:
     [record_features,
      mean_prims_when_in_shadow,      # 4-dim
      mean_prims_when_out_shadow,     # 4-dim
      diff (in - out),                # 4-dim
      std_prims_when_in,              # 4-dim
      std_prims_when_out]             # 4-dim   = F + 20 features
   Meta-classifier (LR + MLP) trained on shadow labels, applied to real
   member / non-member split.

   This detects whether the system's aggregate outputs are sensitive to
   individual record membership — the primary attack surface for DIGIT's
   deterministic executor.

System interface
----------------
Works with any system that either:
  a) Has .generate(query_batch, private_data) → dict with 'primitives'
     (ValidationDIGITModel)
  b) Has .get_primitive_vectors(queries) → (N, 4) int array and a factory
     system_factory(dataset) → new system instance
     (RawBaseline, GenericDPLaplaceBaseline)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from ..data.base import DatasetInfo, PrivateDataset, TabularPrivateDataset
from ..data.query_gen import QueryGenerator

logger = logging.getLogger(__name__)

SEEDS = [0, 1, 2, 3, 4]


# ── Metric helpers ────────────────────────────────────────────────────────────
def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right")
    idx = min(idx, len(tpr) - 1)
    return float(tpr[idx])


def _advantage(auroc: float) -> float:
    return 2.0 * (auroc - 0.5)


def _metrics_from_probs(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    auroc = float(roc_auc_score(y_true, y_score))
    return {
        "auroc":          auroc,
        "advantage":      _advantage(auroc),
        "tpr_at_1pct_fpr": _tpr_at_fpr(y_true, y_score, 0.01),
        "tpr_at_5pct_fpr": _tpr_at_fpr(y_true, y_score, 0.05),
    }


def _chance_metrics() -> Dict[str, float]:
    return {"auroc": 0.5, "advantage": 0.0, "tpr_at_1pct_fpr": 0.0, "tpr_at_5pct_fpr": 0.0}


def _best_classifier(X_tr, y_tr, X_te, y_te, seed: int) -> Optional[Dict[str, float]]:
    """Train LR and MLP, return metrics of the stronger attacker."""
    best = None
    for AttackCls, kw in [
        (LogisticRegression, dict(max_iter=1000, random_state=seed, C=0.1)),
        (MLPClassifier, dict(hidden_layer_sizes=(64, 32), max_iter=500,
                             random_state=seed, early_stopping=True,
                             validation_fraction=0.1)),
    ]:
        try:
            clf = AttackCls(**kw)
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
            m = _metrics_from_probs(y_te, probs)
            if best is None or m["auroc"] > best["auroc"]:
                best = m
        except Exception as exc:
            logger.warning(f"Classifier {AttackCls.__name__} failed: {exc}")
    return best


# ── Primitive vector helpers ──────────────────────────────────────────────────
def _get_primitive_vectors(
    system: Any,
    queries: List[torch.Tensor],
    private_data: PrivateDataset,
) -> np.ndarray:
    """Dispatch to the appropriate system interface → (N, 4) int array."""
    if hasattr(system, "generate"):
        # ValidationDIGITModel: accepts private_data at inference time
        results = []
        for q in queries:
            q_b = q.unsqueeze(0).to(next(system.parameters()).device)
            with torch.no_grad():
                out = system.generate(q_b, private_data)
                p = out["primitives"]
                results.append([
                    int(p.answer_discrete.argmax(-1)[0]),
                    int(p.support_discrete.argmax(-1)[0]),
                    int(p.confidence_discrete.argmax(-1)[0]),
                    int(p.risk_discrete.argmax(-1)[0]),
                ])
        return np.array(results, dtype=np.int64)

    if hasattr(system, "get_primitive_vectors"):
        return system.get_primitive_vectors(queries)

    raise TypeError(f"Unsupported system type: {type(system)}")


def _get_prims_batched(
    system: Any,
    queries: List[torch.Tensor],
    private_data: PrivateDataset,
    batch_size: int = 128,
) -> np.ndarray:
    """Batched version of _get_primitive_vectors for DIGIT models."""
    if not hasattr(system, "generate"):
        return _get_primitive_vectors(system, queries, private_data)

    device = next(system.parameters()).device
    all_prims = []
    for start in range(0, len(queries), batch_size):
        batch = torch.stack(queries[start:start + batch_size]).to(device)
        with torch.no_grad():
            out = system.generate(batch, private_data)
            p = out["primitives"]
            all_prims.append(torch.stack([
                p.answer_discrete.argmax(-1),
                p.support_discrete.argmax(-1),
                p.confidence_discrete.argmax(-1),
                p.risk_discrete.argmax(-1),
            ], dim=1).cpu().numpy())
    return np.concatenate(all_prims, axis=0).astype(np.int64)


# ── Targeting query generation ────────────────────────────────────────────────
def _make_targeting_queries(
    record_features: torch.Tensor,
    n_queries: int,
    n_fields: int,
    seed: int,
) -> List[torch.Tensor]:
    """Generate queries that target a specific record's attribute subgroup.

    Queries vary in specificity (1 to min(8, n_fields) active filters) so
    both broad-group and narrow-group responses are observed.  All active
    filters are set to the record's own attribute values.
    """
    rng = np.random.RandomState(seed)
    queries = []
    for _ in range(n_queries):
        # Bias toward medium specificity (2-5 filters) for best signal
        max_k = min(8, n_fields)
        weights = np.array([1, 3, 4, 4, 3, 2, 1, 1][:max_k], dtype=float)
        weights /= weights.sum()
        k = rng.choice(np.arange(1, max_k + 1), p=weights)
        fields = rng.choice(n_fields, size=k, replace=False)
        q = torch.zeros(n_fields + 1, dtype=torch.long)
        for fi in fields:
            q[fi] = int(record_features[fi].item())
        q[n_fields] = k  # specificity slot
        queries.append(q)
    return queries


# ── Shadow dataset factory ────────────────────────────────────────────────────


# ── Direct attack ─────────────────────────────────────────────────────────────
def _build_direct_features(
    system: Any,
    private_data: PrivateDataset,
    member_features: torch.Tensor,
    nonmember_features: torch.Tensor,
    num_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for the direct MIA (one query per target record)."""
    rng = np.random.RandomState(seed)
    n_each = min(num_samples // 2, member_features.shape[0], nonmember_features.shape[0])
    m_idx  = rng.choice(member_features.shape[0],    size=n_each, replace=False)
    nm_idx = rng.choice(nonmember_features.shape[0], size=n_each, replace=False)
    n_fields = len(private_data.query_fields)

    X, y = [], []
    for feat_bank, indices, label in [
        (member_features, m_idx, 1), (nonmember_features, nm_idx, 0)
    ]:
        queries = []
        for idx in indices:
            rec = feat_bank[idx]
            q = torch.zeros(n_fields + 1, dtype=torch.long)
            for col in range(n_fields):
                q[col] = int(rec[col].item())
            q[n_fields] = int((q[:n_fields] > 0).sum().item())
            queries.append(q)

        prim_vecs = _get_prims_batched(system, queries, private_data)
        for i, idx in enumerate(indices):
            rec_np = feat_bank[idx].cpu().numpy().astype(float)
            X.append(np.concatenate([rec_np, prim_vecs[i].astype(float)]))
            y.append(label)

    return np.array(X), np.array(y)


def run_direct_mia(
    system: Any,
    member_data: PrivateDataset,
    nonmember_data: PrivateDataset,
    seed: int,
    num_samples: int = 1000,
) -> Dict[str, float]:
    """Direct-query MIA (one query per record, single system instance)."""
    X, y = _build_direct_features(
        system, member_data,
        member_data.features, nonmember_data.features,
        num_samples, seed,
    )
    if len(np.unique(y)) < 2:
        out = _chance_metrics()
        out["query_count"] = 0
        out["cumulative_epsilon"] = 0.0
        return out

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    result = _best_classifier(X_tr, y_tr, X_te, y_te, seed)
    out = result if result else _chance_metrics()
    n_each = min(num_samples // 2, member_data.features.shape[0], nonmember_data.features.shape[0])
    out["query_count"] = int(2 * n_each)
    epsilon = float(getattr(system, "epsilon", 0.0)) if hasattr(system, "epsilon") else 0.0
    out["cumulative_epsilon"] = float(out["query_count"] * epsilon)
    return out


# keep old name as alias for runner back-compat
run_mia = run_direct_mia


# ── Shadow-dataset attack ─────────────────────────────────────────────────────
@dataclass
class ShadowMIAConfig:
    """Parameters for the shadow-dataset MIA.

    n_shadow            : K shadow dataset folds (higher = more stable estimate)
    n_queries_per_record: M queries per record per fold (more = better
                          averaging of query noise; ~50 is good)
    shadow_frac         : fraction of combined pool per shadow dataset (0.5 → N/2)
    batch_size          : inference batch size for DIGIT forward passes
    """
    n_shadow: int = 32
    n_queries_per_record: int = 30
    shadow_frac: float = 0.5
    batch_size: int = 128


def _build_shadow_features(
    system: Any,
    member_data: PrivateDataset,
    nonmember_data: PrivateDataset,
    target_member_idx: np.ndarray,
    target_nonmember_idx: np.ndarray,
    cfg: ShadowMIAConfig,
    seed: int,
    system_factory: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for the shadow-dataset meta-classifier.

    For each shadow fold k:
      - Randomly partition combined pool into D_k (in) and D_bar_k (out)
      - For each target record r, issue M targeting queries against D_k
      - Record mean primitive output → shadow_outputs[r, k] shape (4,)

    Then for each record r:
      - in_folds  = folds where r was in D_k
      - out_folds = folds where r was in D_bar_k
      - feature = concat(record_feat, mean(in), mean(out), diff, std(in), std(out))

    Labels: 1 = original member (in member_data), 0 = non-member (in nonmember_data).
    """
    K = cfg.n_shadow
    M = cfg.n_queries_per_record
    n_fields = len(member_data.query_fields)

    # Combined pool: member records first, then non-member
    mem_feats  = member_data.features[target_member_idx]     # (Nm, F)
    nmem_feats = nonmember_data.features[target_nonmember_idx]  # (Nn, F)
    all_feats  = torch.cat([mem_feats, nmem_feats], dim=0)   # (Nm+Nn, F)
    all_labels = torch.cat([
        member_data.labels[target_member_idx],
        nonmember_data.labels[target_nonmember_idx],
    ], dim=0)

    N_total  = all_feats.shape[0]
    N_shadow = max(10, int(N_total * cfg.shadow_frac))
    N_target = N_total

    # shadow_outputs[i, k] = mean primitive vector (shape 4) for record i in fold k
    shadow_outputs = np.zeros((N_target, K, 4), dtype=np.float32)
    # shadow_labels[i, k] = 1 if record i was in D_k
    shadow_labels  = np.zeros((N_target, K), dtype=np.int8)

    logger.info(
        f"  Shadow MIA: K={K} folds, M={M} queries/record, "
        f"N_target={N_target}, N_shadow={N_shadow}"
    )

    for k in range(K):
        fold_seed = seed * 10000 + k
        fold_rng  = np.random.RandomState(fold_seed)

        # Sample shadow dataset indices from the combined pool
        shadow_idx = fold_rng.choice(N_total, size=N_shadow, replace=False)
        shadow_idx_set = set(shadow_idx.tolist())

        # Create shadow private dataset (for executor / baseline)
        # Use member_data as structural template (same schema)
        shadow_ds = _make_combined_shadow_dataset(
            member_data, nonmember_data, shadow_idx,
            target_member_idx, target_nonmember_idx,
        )

        # Build system for this shadow dataset
        if system_factory is not None:
            shadow_system = system_factory(shadow_ds)
        else:
            shadow_system = system  # DIGIT: pass dataset at inference time

        # For each target record, build M targeting queries
        all_queries: List[torch.Tensor] = []
        for i in range(N_target):
            all_queries.extend(_make_targeting_queries(all_feats[i], M, n_fields, fold_seed + i))

        # Batch-evaluate all queries against shadow dataset
        # _get_prims_batched dispatches to .generate(q, dataset) for DIGIT
        # or .get_primitive_vectors(q) for baselines (which use internal dataset)
        prim_all = _get_prims_batched(shadow_system, all_queries, shadow_ds, cfg.batch_size)

        # Aggregate: mean over M queries per record
        prim_all_f = prim_all.astype(np.float32)
        for i in range(N_target):
            start = i * M
            shadow_outputs[i, k] = prim_all_f[start:start + M].mean(axis=0)
            shadow_labels[i, k]  = 1 if i in shadow_idx_set else 0

        if (k + 1) % 8 == 0 or k == K - 1:
            logger.info(f"    fold {k+1}/{K} done")

    # Build per-record attack features
    X, y = [], []
    for i in range(N_target):
        in_folds  = np.where(shadow_labels[i] == 1)[0]
        out_folds = np.where(shadow_labels[i] == 0)[0]

        if len(in_folds) == 0 or len(out_folds) == 0:
            continue  # degenerate — skip

        in_mean  = shadow_outputs[i, in_folds].mean(axis=0)   # (4,)
        out_mean = shadow_outputs[i, out_folds].mean(axis=0)  # (4,)
        in_std   = shadow_outputs[i, in_folds].std(axis=0)    # (4,)
        out_std  = shadow_outputs[i, out_folds].std(axis=0)   # (4,)
        diff     = in_mean - out_mean                          # (4,)

        rec_np = all_feats[i].cpu().numpy().astype(float)
        feat = np.concatenate([rec_np, in_mean, out_mean, diff, in_std, out_std])
        X.append(feat)
        # Label: 1 if this record is an original member, 0 if non-member
        y.append(1 if i < len(target_member_idx) else 0)

    return np.array(X), np.array(y)


def _make_combined_shadow_dataset(
    member_data: PrivateDataset,
    nonmember_data: PrivateDataset,
    shadow_indices: np.ndarray,   # indices into combined [member | nonmember] pool
    target_member_idx: np.ndarray,
    target_nonmember_idx: np.ndarray,
) -> TabularPrivateDataset:
    """Create a shadow dataset from the combined pool using shadow_indices."""
    all_feats  = torch.cat([
        member_data.features[target_member_idx],
        nonmember_data.features[target_nonmember_idx],
    ], dim=0)
    all_labels = torch.cat([
        member_data.labels[target_member_idx],
        nonmember_data.labels[target_nonmember_idx],
    ], dim=0)

    idx_t    = torch.tensor(shadow_indices, dtype=torch.long)
    sub_feat = all_feats[idx_t]
    sub_lbl  = all_labels[idx_t]
    n        = len(shadow_indices)
    pos_rate = float(sub_lbl.float().mean()) if n > 0 else 0.0

    info = member_data.info
    sub_info = DatasetInfo(
        dataset_name=info.dataset_name,
        split_name="shadow",
        n_records=n,
        n_features=info.n_features,
        label_name=info.label_name,
        positive_rate=pos_rate,
        query_fields=info.query_fields,
        field_vocab_sizes=info.field_vocab_sizes,
    )
    return TabularPrivateDataset(sub_feat, sub_lbl, member_data.query_fields, sub_info)


def run_shadow_mia(
    system: Any,
    member_data: PrivateDataset,
    nonmember_data: PrivateDataset,
    seed: int,
    cfg: Optional[ShadowMIAConfig] = None,
    num_samples: int = 500,
    system_factory: Optional[Callable] = None,
) -> Dict[str, float]:
    """Run the shadow-dataset MIA for one seed.

    Parameters
    ----------
    system         : trained system (ValidationDIGITModel, RawBaseline, etc.)
    member_data    : training-set records (members)
    nonmember_data : test-set records (non-members)
    seed           : RNG seed
    cfg            : ShadowMIAConfig (defaults used if None)
    num_samples    : number of target records (half members, half non-members)
    system_factory : callable(dataset) → system instance; required for
                     baselines (RawBaseline, GenericDPLaplaceBaseline).
                     None → DIGIT (passes dataset at inference time).
    """
    if cfg is None:
        cfg = ShadowMIAConfig()

    rng = np.random.RandomState(seed)
    n_each = min(
        num_samples // 2,
        member_data.num_records,
        nonmember_data.num_records,
    )
    target_member_idx    = rng.choice(member_data.num_records,    size=n_each, replace=False)
    target_nonmember_idx = rng.choice(nonmember_data.num_records, size=n_each, replace=False)

    X, y = _build_shadow_features(
        system, member_data, nonmember_data,
        target_member_idx, target_nonmember_idx,
        cfg, seed, system_factory,
    )

    if len(np.unique(y)) < 2 or len(X) < 20:
        logger.warning("Shadow MIA: insufficient class diversity — returning chance metrics.")
        out = _chance_metrics()
        out["query_count"] = 0
        out["cumulative_epsilon"] = 0.0
        return out

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y,
    )
    result = _best_classifier(X_tr, y_tr, X_te, y_te, seed)
    out = result if result else _chance_metrics()
    epsilon = float(getattr(system, "epsilon", 0.0)) if hasattr(system, "epsilon") else 0.0
    n_each = min(
        num_samples // 2,
        member_data.num_records,
        nonmember_data.num_records,
    )
    n_target = 2 * n_each
    out["query_count"] = int(cfg.n_shadow * cfg.n_queries_per_record * n_target)
    out["cumulative_epsilon"] = float(out["query_count"] * epsilon)
    return out


# ── Combined run (direct + shadow, take stronger) ────────────────────────────
def run_mia_full(
    system: Any,
    member_data: PrivateDataset,
    nonmember_data: PrivateDataset,
    seed: int,
    num_samples: int = 1000,
    shadow_cfg: Optional[ShadowMIAConfig] = None,
    system_factory: Optional[Callable] = None,
) -> Dict[str, float]:
    """Run both direct and shadow attacks; return the stronger result.

    The returned dict includes both sub-results for transparency:
      result["direct"]  : direct attack metrics
      result["shadow"]  : shadow attack metrics
      result["auroc"]   : max(direct, shadow) — the harness-reported number
      result["attack"]  : "direct" | "shadow" — which was stronger
    """
    logger.info("    Running direct attack …")
    direct = run_direct_mia(system, member_data, nonmember_data, seed, num_samples)

    logger.info(f"    Direct: AUROC={direct['auroc']:.3f}")

    n_shadow_samples = min(num_samples, 500)  # shadow is slower; cap target records
    logger.info(f"    Running shadow attack (n_target={n_shadow_samples}) …")
    shadow = run_shadow_mia(
        system, member_data, nonmember_data, seed,
        cfg=shadow_cfg,
        num_samples=n_shadow_samples,
        system_factory=system_factory,
    )
    logger.info(f"    Shadow: AUROC={shadow['auroc']:.3f}")

    # Take the stronger attack (worst-case adversary)
    if shadow["auroc"] >= direct["auroc"]:
        best = dict(shadow)
        best["attack"] = "shadow"
    else:
        best = dict(direct)
        best["attack"] = "direct"

    best["direct"] = direct
    best["shadow"] = shadow
    best["query_count"] = int(direct.get("query_count", 0) + shadow.get("query_count", 0))
    best["cumulative_epsilon"] = float(
        direct.get("cumulative_epsilon", 0.0) + shadow.get("cumulative_epsilon", 0.0)
    )
    return best


def run_mia_all_seeds(
    system: Any,
    member_data: PrivateDataset,
    nonmember_data: PrivateDataset,
    seeds: List[int] = SEEDS,
    num_samples: int = 1000,
    system_key: str = "unknown",
    shadow_cfg: Optional[ShadowMIAConfig] = None,
    system_factory: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run MIA across all seeds; return aggregated results."""
    per_seed = []
    for s in seeds:
        logger.info(f"  MIA seed={s} system={system_key}")
        per_seed.append(run_mia_full(
            system, member_data, nonmember_data, s,
            num_samples=num_samples,
            shadow_cfg=shadow_cfg,
            system_factory=system_factory,
        ))

    metrics = ["auroc", "advantage", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"]
    agg = {"system_key": system_key, "per_seed": per_seed}
    for m in metrics:
        vals = [r[m] for r in per_seed]
        agg[f"{m}_mean"] = float(np.mean(vals))
        agg[f"{m}_std"]  = float(np.std(vals))
        agg[f"{m}_values"] = vals

    logger.info(
        f"  [{system_key}] AUROC = {agg['auroc_mean']:.3f} ± {agg['auroc_std']:.3f} | "
        f"Adv = {agg['advantage_mean']:.3f} | "
        f"TPR@1% = {agg['tpr_at_1pct_fpr_mean']:.3f} | "
        f"TPR@5% = {agg['tpr_at_5pct_fpr_mean']:.3f}"
    )
    return agg
