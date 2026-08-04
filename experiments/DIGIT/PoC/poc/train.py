"""Training loop for DIGIT with multi-seed support and early stopping."""

from __future__ import annotations

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .config import Config
from .data.vocabulary import Vocabulary
from .data.ground_truth import ANSWER_LABELS, SUPPORT_LABELS
from .data.dataset import AdultIntuitionDataset, collate_fn
from .data.private_store import PrivateAdultDataset
from .models.digit import DIGITModel
from .models.baselines import BaselineA, BaselineB
from .losses import DIGITLoss

logger = logging.getLogger(__name__)


def get_tau(step: int, config: Config) -> float:
    """Anneal Gumbel temperature."""
    if step >= config.gumbel_anneal_steps:
        return config.gumbel_tau_end
    ratio = step / config.gumbel_anneal_steps
    return config.gumbel_tau_start + (config.gumbel_tau_end - config.gumbel_tau_start) * ratio


def evaluate_baselines(
    config: Config,
    vocab: Vocabulary,
    test_dataset: AdultIntuitionDataset,
    private_data: PrivateAdultDataset,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate Baseline A and B."""
    results = {}
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    # Baseline A
    baseline_a = BaselineA(config, vocab).to(device)
    a_correct_3class = 0
    a_total = 0
    for queries, prim_targets, target_ids in test_loader:
        queries = queries.to(device)
        out = baseline_a(queries, private_data)
        gt_3class = prim_targets[:, 0].clone()
        gt_3class[gt_3class >= 2] = 2
        a_correct_3class += (out["answers"].cpu() == gt_3class).sum().item()
        a_total += queries.size(0)

    results["baseline_a"] = {
        "accuracy_3class": a_correct_3class / max(a_total, 1),
    }

    # Baseline B
    baseline_b = BaselineB(config, vocab).to(device)
    b_answer_correct = 0
    b_support_correct = 0
    b_confidence_correct = 0
    b_risk_correct = 0
    b_total = 0
    for queries, prim_targets, target_ids in test_loader:
        queries = queries.to(device)
        out = baseline_b(queries, private_data)
        prims = out["primitives"]
        b_answer_correct += (prims["answer"].cpu() == prim_targets[:, 0]).sum().item()
        b_support_correct += (prims["support"].cpu() == prim_targets[:, 1]).sum().item()
        b_confidence_correct += (prims["confidence"].cpu() == prim_targets[:, 2]).sum().item()
        b_risk_correct += (prims["risk"].cpu() == prim_targets[:, 3]).sum().item()
        b_total += queries.size(0)

    results["baseline_b"] = {
        "answer_accuracy": b_answer_correct / max(b_total, 1),
        "support_accuracy": b_support_correct / max(b_total, 1),
        "confidence_accuracy": b_confidence_correct / max(b_total, 1),
        "risk_accuracy": b_risk_correct / max(b_total, 1),
    }

    return results


def train_single_seed(
    config: Config,
    vocab: Vocabulary,
    train_dataset: AdultIntuitionDataset,
    val_dataset: AdultIntuitionDataset,
    private_data: PrivateAdultDataset,
    device: torch.device,
    seed: int,
    output_dir: str,
) -> Dict[str, Any]:
    """Train DIGIT for a single seed."""

    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = DIGITModel(config, vocab).to(device)
    class_weights = train_dataset.get_class_weights()
    class_weights = {k: v.to(device) for k, v in class_weights.items()}
    criterion = DIGITLoss(config, pad_idx=vocab.pad_idx,
                          class_weights=class_weights).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Seed {seed}] Parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate,
                      weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.num_epochs // 5 + 1
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in
                        ["total", "primitive", "generation", "policy",
                         "leakage", "abstention"]}
        epoch_correct = {k: 0 for k in ["answer", "support", "confidence", "risk"]}
        epoch_total = 0
        t0 = time.time()

        for queries, prim_targets, target_ids in train_loader:
            queries = queries.to(device)
            prim_targets = prim_targets.to(device)
            target_ids = target_ids.to(device)

            tau = get_tau(global_step, config)

            out = model(queries, private_data, target_ids,
                        bottleneck_mode="gumbel", tau=tau)

            losses = criterion(
                decoder_logits=out["decoder_logits"],
                primitives=out["primitives"],
                target_ids=target_ids,
                prim_targets=prim_targets,
                executor_features=out["executor_features"],
            )

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            B = queries.size(0)
            epoch_total += B
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item() * B

            prims = out["primitives"]
            epoch_correct["answer"] += (prims.answer_logits.argmax(-1) == prim_targets[:, 0]).sum().item()
            epoch_correct["support"] += (prims.support_logits.argmax(-1) == prim_targets[:, 1]).sum().item()
            epoch_correct["confidence"] += (prims.confidence_logits.argmax(-1) == prim_targets[:, 2]).sum().item()
            epoch_correct["risk"] += (prims.risk_logits.argmax(-1) == prim_targets[:, 3]).sum().item()

            global_step += 1

        scheduler.step()

        for k in epoch_losses:
            epoch_losses[k] /= max(epoch_total, 1)
        epoch_acc = {k: v / max(epoch_total, 1) for k, v in epoch_correct.items()}

        # Validation
        model.eval()
        val_losses = {k: 0.0 for k in epoch_losses}
        val_correct = {k: 0 for k in epoch_correct}
        val_total = 0

        with torch.no_grad():
            for queries, prim_targets, target_ids in val_loader:
                queries = queries.to(device)
                prim_targets = prim_targets.to(device)
                target_ids = target_ids.to(device)

                out = model(queries, private_data, target_ids,
                            bottleneck_mode="hard", tau=1.0)
                losses = criterion(
                    decoder_logits=out["decoder_logits"],
                    primitives=out["primitives"],
                    target_ids=target_ids,
                    prim_targets=prim_targets,
                    executor_features=out["executor_features"],
                )

                B = queries.size(0)
                val_total += B
                for k in val_losses:
                    val_losses[k] += losses[k].item() * B

                prims = out["primitives"]
                val_correct["answer"] += (prims.answer_logits.argmax(-1) == prim_targets[:, 0]).sum().item()
                val_correct["support"] += (prims.support_logits.argmax(-1) == prim_targets[:, 1]).sum().item()
                val_correct["confidence"] += (prims.confidence_logits.argmax(-1) == prim_targets[:, 2]).sum().item()
                val_correct["risk"] += (prims.risk_logits.argmax(-1) == prim_targets[:, 3]).sum().item()

        for k in val_losses:
            val_losses[k] /= max(val_total, 1)
        val_acc = {k: v / max(val_total, 1) for k, v in val_correct.items()}

        elapsed = time.time() - t0

        logger.info(
            f"[Seed {seed}] Epoch {epoch+1}/{config.num_epochs} ({elapsed:.1f}s) | "
            f"Train loss: {epoch_losses['total']:.4f} | "
            f"Val loss: {val_losses['total']:.4f} | "
            f"Ans acc: {val_acc['answer']:.3f} | "
            f"Sup acc: {val_acc['support']:.3f} | "
            f"Tau: {get_tau(global_step, config):.3f}"
        )

        # Early stopping
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(model.state_dict(), os.path.join(seed_dir, "best_model.pt"))
            patience_counter = 0
            logger.info(f"  -> New best (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"  -> Early stopping at epoch {epoch+1}")
                break

        history.append({
            "epoch": epoch + 1,
            "train_losses": epoch_losses,
            "train_acc": epoch_acc,
            "val_losses": val_losses,
            "val_acc": val_acc,
            "tau": get_tau(global_step, config),
            "lr": optimizer.param_groups[0]["lr"],
        })

    # Save final
    torch.save(model.state_dict(), os.path.join(seed_dir, "final_model.pt"))
    with open(os.path.join(seed_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Generate samples from best model
    model.load_state_dict(torch.load(os.path.join(seed_dir, "best_model.pt"),
                                      weights_only=True))
    model.eval()
    sample_outputs = []
    for i in range(min(10, len(val_dataset))):
        q = val_dataset.queries[i].unsqueeze(0).to(device)
        result = model.generate(q, private_data)
        text = vocab.decode(result["token_ids"][0].cpu().tolist())
        prim_idx = model.bottleneck.get_primitive_indices(result["primitives"])
        sample_outputs.append({
            "answer": ANSWER_LABELS[prim_idx["answer"][0].item()],
            "support": SUPPORT_LABELS[prim_idx["support"][0].item()],
            "generated_text": text,
            "ground_truth": val_dataset.ground_truths[i]["response_text"],
        })

    return {
        "seed": seed,
        "history": history,
        "best_val_loss": best_val_loss,
        "sample_outputs": sample_outputs,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def evaluate_on_test(
    config: Config,
    vocab: Vocabulary,
    test_dataset: AdultIntuitionDataset,
    private_data: PrivateAdultDataset,
    model_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a trained model on the test set."""
    model = DIGITModel(config, vocab).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    correct = {k: 0 for k in ["answer", "support", "confidence", "risk"]}
    joint_correct = 0
    total = 0
    leakage_sum = 0.0

    all_answer_preds = []
    all_answer_targets = []

    with torch.no_grad():
        for queries, prim_targets, target_ids in test_loader:
            queries = queries.to(device)
            prim_targets = prim_targets.to(device)
            target_ids = target_ids.to(device)

            out = model(queries, private_data, target_ids,
                        bottleneck_mode="hard", tau=1.0)
            prims = out["primitives"]

            B = queries.size(0)
            total += B

            ans_pred = prims.answer_logits.argmax(-1)
            sup_pred = prims.support_logits.argmax(-1)
            conf_pred = prims.confidence_logits.argmax(-1)
            risk_pred = prims.risk_logits.argmax(-1)

            correct["answer"] += (ans_pred == prim_targets[:, 0]).sum().item()
            correct["support"] += (sup_pred == prim_targets[:, 1]).sum().item()
            correct["confidence"] += (conf_pred == prim_targets[:, 2]).sum().item()
            correct["risk"] += (risk_pred == prim_targets[:, 3]).sum().item()

            joint = (
                (ans_pred == prim_targets[:, 0])
                & (sup_pred == prim_targets[:, 1])
                & (conf_pred == prim_targets[:, 2])
                & (risk_pred == prim_targets[:, 3])
            )
            joint_correct += joint.sum().item()

            all_answer_preds.extend(ans_pred.cpu().tolist())
            all_answer_targets.extend(prim_targets[:, 0].cpu().tolist())

            # Leakage
            dec_summary = out["decoder_logits"].mean(dim=1)
            k = min(32, dec_summary.size(-1))
            d = dec_summary[:, :k]
            e = out["executor_features"]
            d = d - d.mean(dim=0, keepdim=True)
            e = e - e.mean(dim=0, keepdim=True)
            d_std = d.std(dim=0, keepdim=True).clamp(min=1e-6)
            e_std = e.std(dim=0, keepdim=True).clamp(min=1e-6)
            corr = ((d / d_std).T @ (e / e_std)) / max(B, 1)
            leakage_sum += (corr ** 2).mean().item() * B

    acc = {k: v / max(total, 1) for k, v in correct.items()}
    acc["joint"] = joint_correct / max(total, 1)
    acc["leakage"] = leakage_sum / max(total, 1)

    return {
        "accuracy": acc,
        "predictions": all_answer_preds,
        "targets": all_answer_targets,
        "num_samples": total,
    }
