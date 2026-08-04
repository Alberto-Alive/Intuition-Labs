"""Training loop for all three systems (A, B, C).

System A: BaselineA — deterministic executor, 3-class rule, template decoder
System B: BaselineB — deterministic executor, full rule-based primitives, template decoder
System C: IntuitionModel — learned encoder, bottleneck, decoder
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .config import Config
from .data.dataset import IntuitionDataset, PrivateDataset, collate_fn
from .data.vocabulary import Vocabulary, ANSWER_LABELS, SUPPORT_LABELS
from .models.full_model import IntuitionModel
from .models.baselines import BaselineA, BaselineB
from .losses import IntuitionLoss

logger = logging.getLogger(__name__)


def get_tau(step: int, config: Config) -> float:
    """Anneal Gumbel temperature from tau_start to tau_end."""
    if step >= config.gumbel_anneal_steps:
        return config.gumbel_tau_end
    ratio = step / config.gumbel_anneal_steps
    return config.gumbel_tau_start + (config.gumbel_tau_end - config.gumbel_tau_start) * ratio


def evaluate_baselines(
    config: Config,
    vocab: Vocabulary,
    val_dataset: IntuitionDataset,
    private_data: PrivateDataset,
    device: torch.device,
) -> Dict[str, Any]:
    """Run baselines A and B on validation data."""
    results = {}

    baseline_a = BaselineA(config, vocab).to(device)
    baseline_b = BaselineB(config, vocab).to(device)

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    # Evaluate baseline A
    a_correct_3class = 0
    a_total = 0
    for queries, group_ids, prim_targets, target_ids in val_loader:
        queries = queries.to(device)
        group_ids = group_ids.to(device)
        out = baseline_a(queries, group_ids, private_data)
        # Map ground truth 6-class to 3-class: 0->0(YES), 1->1(NO), 2-5->2(MAYBE)
        gt_3class = prim_targets[:, 0].clone()
        gt_3class[gt_3class >= 2] = 2
        a_correct_3class += (out["answers"].cpu() == gt_3class).sum().item()
        a_total += queries.size(0)

    results["baseline_a"] = {
        "accuracy_3class": a_correct_3class / max(a_total, 1),
        "sample_outputs": [],
    }

    # Collect sample outputs from baseline A
    sample_q = val_dataset.queries[0].unsqueeze(0).to(device)
    sample_g = torch.tensor([val_dataset.query_group_ids[0]], device=device)
    out_a = baseline_a(sample_q, sample_g, private_data)
    results["baseline_a"]["sample_outputs"].append(out_a["texts"][0])

    # Evaluate baseline B
    b_answer_correct = 0
    b_support_correct = 0
    b_total = 0
    for queries, group_ids, prim_targets, target_ids in val_loader:
        queries = queries.to(device)
        group_ids = group_ids.to(device)
        out = baseline_b(queries, group_ids, private_data)
        prims = out["primitives"]
        b_answer_correct += (prims["answer"].cpu() == prim_targets[:, 0]).sum().item()
        b_support_correct += (prims["support"].cpu() == prim_targets[:, 1]).sum().item()
        b_total += queries.size(0)

    results["baseline_b"] = {
        "answer_accuracy": b_answer_correct / max(b_total, 1),
        "support_accuracy": b_support_correct / max(b_total, 1),
        "sample_outputs": [],
    }

    out_b = baseline_b(sample_q, sample_g, private_data)
    results["baseline_b"]["sample_outputs"].append(out_b["texts"][0])

    return results


def train_model_c(
    config: Config,
    vocab: Vocabulary,
    train_dataset: IntuitionDataset,
    val_dataset: IntuitionDataset,
    private_data: PrivateDataset,
    device: torch.device,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Train the full IntuitionModel (System C)."""

    os.makedirs(output_dir, exist_ok=True)

    model = IntuitionModel(config, vocab).to(device)
    criterion = IntuitionLoss(config, pad_idx=vocab.pad_idx).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.num_epochs // 5 + 1)

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
    history = []

    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = {
            "total": 0, "primitive": 0, "generation": 0,
            "policy": 0, "leakage": 0, "abstention": 0,
        }
        epoch_correct = {"answer": 0, "support": 0, "confidence": 0, "risk": 0}
        epoch_total = 0

        t0 = time.time()

        for batch_idx, (queries, group_ids, prim_targets, target_ids) in enumerate(train_loader):
            queries = queries.to(device)
            group_ids = group_ids.to(device)
            prim_targets = prim_targets.to(device)
            target_ids = target_ids.to(device)

            tau = get_tau(global_step, config)

            # Forward
            out = model(
                queries, group_ids, private_data, target_ids,
                bottleneck_mode="gumbel", tau=tau,
            )

            # Loss
            losses = criterion(
                decoder_logits=out["decoder_logits"],
                primitives=out["primitives"],
                target_ids=target_ids,
                prim_targets=prim_targets,
                executor_features=out["executor_features"],
            )

            # Backward
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # Accumulate metrics
            B = queries.size(0)
            epoch_total += B
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item() * B

            # Primitive accuracy
            prims = out["primitives"]
            epoch_correct["answer"] += (prims.answer_logits.argmax(-1) == prim_targets[:, 0]).sum().item()
            epoch_correct["support"] += (prims.support_logits.argmax(-1) == prim_targets[:, 1]).sum().item()
            epoch_correct["confidence"] += (prims.confidence_logits.argmax(-1) == prim_targets[:, 2]).sum().item()
            epoch_correct["risk"] += (prims.risk_logits.argmax(-1) == prim_targets[:, 3]).sum().item()

            global_step += 1

        scheduler.step()

        # Average training metrics
        for k in epoch_losses:
            epoch_losses[k] /= max(epoch_total, 1)
        epoch_acc = {k: v / max(epoch_total, 1) for k, v in epoch_correct.items()}

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_losses = {
            "total": 0, "primitive": 0, "generation": 0,
            "policy": 0, "leakage": 0, "abstention": 0,
        }
        val_correct = {"answer": 0, "support": 0, "confidence": 0, "risk": 0}
        val_total = 0

        with torch.no_grad():
            for queries, group_ids, prim_targets, target_ids in val_loader:
                queries = queries.to(device)
                group_ids = group_ids.to(device)
                prim_targets = prim_targets.to(device)
                target_ids = target_ids.to(device)

                out = model(
                    queries, group_ids, private_data, target_ids,
                    bottleneck_mode="hard", tau=1.0,
                )

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

        # Log
        logger.info(
            f"Epoch {epoch+1}/{config.num_epochs} ({elapsed:.1f}s) | "
            f"Train loss: {epoch_losses['total']:.4f} | "
            f"Val loss: {val_losses['total']:.4f} | "
            f"Answer acc: {val_acc['answer']:.3f} | "
            f"Support acc: {val_acc['support']:.3f} | "
            f"Tau: {get_tau(global_step, config):.3f}"
        )

        # Save best
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info(f"  -> New best model saved (val loss: {best_val_loss:.4f})")

        history.append({
            "epoch": epoch + 1,
            "train_losses": epoch_losses,
            "train_acc": epoch_acc,
            "val_losses": val_losses,
            "val_acc": val_acc,
            "tau": get_tau(global_step, config),
            "lr": optimizer.param_groups[0]["lr"],
        })

    # Save final model and history
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Generate sample outputs ───────────────────────────────────────
    model.eval()
    sample_outputs = []
    for i in range(min(5, len(val_dataset))):
        q = val_dataset.queries[i].unsqueeze(0).to(device)
        g = torch.tensor([val_dataset.query_group_ids[i]], device=device)
        result = model.generate(q, g, private_data)

        text = vocab.decode(result["token_ids"][0].cpu().tolist())
        prim_idx = model.bottleneck.get_primitive_indices(result["primitives"])

        sample_outputs.append({
            "answer": ANSWER_LABELS[prim_idx["answer"][0].item()],
            "support": SUPPORT_LABELS[prim_idx["support"][0].item()],
            "generated_text": text,
            "ground_truth": val_dataset.target_texts[i],
        })

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "sample_outputs": sample_outputs,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def run_experiment(config_path: Optional[str] = None, output_dir: str = "outputs"):
    """Run the full experiment: baselines + main model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    torch.manual_seed(config.seed)

    # Build vocabulary
    vocab = Vocabulary()
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Build private dataset
    private_data = PrivateDataset(
        num_records=config.num_private_records,
        num_features=config.num_query_fields,
    )
    logger.info(f"Private dataset: {private_data.num_records} records, {private_data.num_groups} groups")

    # Build train/val datasets
    train_dataset = IntuitionDataset(
        num_samples=config.num_train_samples,
        num_query_fields=config.num_query_fields,
        vocab=vocab,
        max_output_len=config.max_output_len,
        seed=config.seed,
    )
    val_dataset = IntuitionDataset(
        num_samples=config.num_val_samples,
        num_query_fields=config.num_query_fields,
        vocab=vocab,
        max_output_len=config.max_output_len,
        seed=config.seed + 1,
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Baselines ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating Baseline A (3-class rule + template)")
    logger.info("=" * 60)
    baseline_results = evaluate_baselines(
        config, vocab, val_dataset, private_data, device
    )
    logger.info(f"Baseline A — 3-class accuracy: {baseline_results['baseline_a']['accuracy_3class']:.3f}")
    logger.info(f"  Sample: {baseline_results['baseline_a']['sample_outputs'][0]}")

    logger.info("=" * 60)
    logger.info("Evaluating Baseline B (full rule-based + template)")
    logger.info("=" * 60)
    logger.info(f"Baseline B — answer accuracy: {baseline_results['baseline_b']['answer_accuracy']:.3f}")
    logger.info(f"Baseline B — support accuracy: {baseline_results['baseline_b']['support_accuracy']:.3f}")
    logger.info(f"  Sample: {baseline_results['baseline_b']['sample_outputs'][0]}")

    # ── Main Model (System C) ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training System C (learned encoder + bottleneck + decoder)")
    logger.info("=" * 60)
    model_c_results = train_model_c(
        config, vocab, train_dataset, val_dataset,
        private_data, device, output_dir,
    )

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline A (YES/NO/MAYBE only):  3-class acc = {baseline_results['baseline_a']['accuracy_3class']:.3f}")
    logger.info(f"Baseline B (rich primitives):     answer acc  = {baseline_results['baseline_b']['answer_accuracy']:.3f}")
    logger.info(f"                                  support acc = {baseline_results['baseline_b']['support_accuracy']:.3f}")

    final = model_c_results["history"][-1] if model_c_results["history"] else {}
    if final:
        logger.info(f"System C (learned):              answer acc  = {final['val_acc']['answer']:.3f}")
        logger.info(f"                                  support acc = {final['val_acc']['support']:.3f}")
        logger.info(f"                                  val loss    = {final['val_losses']['total']:.4f}")
        logger.info(f"                                  leakage     = {final['val_losses']['leakage']:.4f}")

    logger.info(f"\nModel size: {model_c_results['trainable_params']:,} trainable parameters")
    logger.info(f"Best val loss: {model_c_results['best_val_loss']:.4f}")

    logger.info("\nSample generated outputs (System C):")
    for s in model_c_results["sample_outputs"]:
        logger.info(f"  [{s['answer']}|{s['support']}] {s['generated_text']}")
        logger.info(f"    GT: {s['ground_truth']}")

    # Save all results
    all_results = {
        "baselines": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in baseline_results.items()
        },
        "model_c": {
            "best_val_loss": model_c_results["best_val_loss"],
            "trainable_params": model_c_results["trainable_params"],
            "sample_outputs": model_c_results["sample_outputs"],
        },
    }
    with open(os.path.join(output_dir, "experiment_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}/")
    return all_results
