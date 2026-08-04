from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from typing import Dict, Iterable, List

from src.experiments.architecture_search import StageConfig, _candidate_specs, _coordinator_config


STAGE7_BASE_LR = 0.0003
STAGE7_BASE_CLIP = 1.0
STAGE7_BASE_EPOCHS = 50
STAGE7_BASE_PATIENCE = 6
STAGE7_ACTIVE_MESSAGE_LAYERS = (-4, -3, -2, -1)


@dataclass(frozen=True)
class Stage7VariantSpec:
    name: str
    family: str
    description: str
    transport_config: Dict[str, object]
    refinement_steps: int
    workspace_slots: int = 0
    lr: float = STAGE7_BASE_LR
    epochs: int = STAGE7_BASE_EPOCHS
    patience: int = STAGE7_BASE_PATIENCE
    gradient_clip_norm: float = STAGE7_BASE_CLIP
    weight_decay: float = 0.0001
    dropout: float = 0.0
    model_dim_multiplier: float = 1.0

    def to_candidate(self, stage: StageConfig):
        locked = next(candidate for candidate in _candidate_specs() if candidate.name == "topk_attention_no_head")
        msg = replace(
            locked.message_config,
            readout_source="pooled",
            use_message_head=False,
            use_private_cue_aux=False,
            aux_loss_weight=0.0,
            coordinator_family="latent_evidence_transport",
            active_message_layers=STAGE7_ACTIVE_MESSAGE_LAYERS,
            canonicalize_role_order=True,
        )
        base = _coordinator_config(stage, family="latent_evidence_transport", num_layers=1, dropout=float(self.dropout))
        coord = replace(
            base,
            family="latent_evidence_transport",
            model_dim=max(32, int(round(base.model_dim * float(self.model_dim_multiplier)))),
            transport_config=dict(self.transport_config),
        )
        return SimpleNamespace(
            name=self.name,
            description=self.description,
            message_config=msg,
            coordinator_config=coord,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def stage7_variant_plan(max_variants: int = 0) -> List[Stage7VariantSpec]:
    variants: List[Stage7VariantSpec] = []

    def add(
        name: str,
        family: str,
        description: str,
        refinement_steps: int,
        *,
        workspace_slots: int = 0,
        model_dim_multiplier: float = 1.0,
        **transport: object,
    ) -> None:
        config = {
            "family": family,
            "refinement_steps": int(refinement_steps),
            "workspace_slots": int(workspace_slots),
            **transport,
        }
        variants.append(
            Stage7VariantSpec(
                name=name,
                family=family,
                description=description,
                transport_config=config,
                refinement_steps=int(refinement_steps),
                workspace_slots=int(workspace_slots),
                model_dim_multiplier=float(model_dim_multiplier),
            )
        )

    for steps in (1, 2, 4, 6):
        add(
            f"a_candidate_only_t{steps}_shared_gated_pre",
            "A_candidate_only_refinement",
            "Fixed evidence; candidates refine by repeatedly attending to role/avenue token evidence with shared gated pre-norm blocks.",
            steps,
            shared_update=True,
            gated_residual=True,
            norm_position="pre",
            candidate_reads_evidence=True,
            update_evidence=False,
        )
    add(
        "a_candidate_only_t4_unshared_gated_pre",
        "A_candidate_only_refinement",
        "Fixed evidence; candidates refine for four unshared gated pre-norm steps.",
        4,
        shared_update=False,
        gated_residual=True,
        norm_position="pre",
        candidate_reads_evidence=True,
        update_evidence=False,
    )
    add(
        "a_candidate_only_t2_shared_plain_post",
        "A_candidate_only_refinement",
        "Fixed evidence; candidates refine with plain residual post-norm updates.",
        2,
        shared_update=True,
        gated_residual=False,
        norm_position="post",
        candidate_reads_evidence=True,
        update_evidence=False,
    )

    for steps in (2, 4):
        add(
            f"b_bidirectional_t{steps}_shared",
            "B_bidirectional_evidence_transport",
            "Candidates attend to evidence, evidence attends back to candidates, then candidates read modified evidence.",
            steps,
            shared_update=True,
            gated_residual=True,
            norm_position="pre",
            candidate_reads_evidence=True,
            update_evidence=True,
            stop_gradient_evidence_update=False,
        )
    add(
        "b_bidirectional_t2_stopgrad_diag",
        "B_bidirectional_evidence_transport",
        "Bidirectional evidence update with stop-gradient through candidate-to-evidence diagnostic path.",
        2,
        shared_update=True,
        gated_residual=True,
        norm_position="pre",
        candidate_reads_evidence=True,
        update_evidence=True,
        stop_gradient_evidence_update=True,
    )
    add(
        "b_bidirectional_t4_unshared",
        "B_bidirectional_evidence_transport",
        "Bidirectional evidence transport with unshared candidate/evidence update blocks.",
        4,
        shared_update=False,
        gated_residual=True,
        norm_position="pre",
        candidate_reads_evidence=True,
        update_evidence=True,
        share_candidate_evidence_weights=False,
    )

    for slots in (4, 8, 16):
        add(
            f"c_workspace_learned_m{slots}_t2",
            "C_latent_workspace",
            "Learned workspace slots aggregate evidence; candidates read workspace and evidence.",
            2,
            workspace_slots=slots,
            workspace_init="learned",
            evidence_writes_workspace=True,
            candidate_reads_workspace=True,
            candidate_conditioned_workspace_read=True,
            candidate_writes_workspace=False,
            candidate_reads_evidence=True,
            update_evidence=False,
        )
    add(
        "c_workspace_evidencepooled_m8_t2",
        "C_latent_workspace",
        "Workspace slots initialized from pooled evidence, then used for candidate-conditioned readout.",
        2,
        workspace_slots=8,
        workspace_init="evidence_pooled",
        evidence_writes_workspace=True,
        candidate_reads_workspace=True,
        candidate_conditioned_workspace_read=True,
        candidate_writes_workspace=False,
        candidate_reads_evidence=True,
        update_evidence=False,
    )
    add(
        "c_workspace_candidate_write_m8_t2",
        "C_latent_workspace",
        "Evidence and candidates both write into learned workspace slots before candidate readout.",
        2,
        workspace_slots=8,
        workspace_init="learned",
        evidence_writes_workspace=True,
        candidate_reads_workspace=True,
        candidate_conditioned_workspace_read=True,
        candidate_writes_workspace=True,
        candidate_reads_evidence=True,
        update_evidence=False,
    )

    for weight in (0.05, 0.10, 0.20):
        suffix = str(weight).replace(".", "p")
        add(
            f"d_energy_margin_t2_w{suffix}",
            "D_energy_margin_refinement",
            "Candidate-only transport with auxiliary margin-improvement trajectory loss.",
            2,
            shared_update=True,
            gated_residual=True,
            norm_position="pre",
            candidate_reads_evidence=True,
            update_evidence=False,
            aux_loss="margin_improvement",
            aux_weight=weight,
            aux_margin=0.05,
            energy_source="candidate_state",
        )
    add(
        "d_energy_compat_t2_w0p10",
        "D_energy_margin_refinement",
        "Energy trajectory loss using candidate-evidence compatibility logits.",
        2,
        shared_update=True,
        gated_residual=True,
        norm_position="pre",
        candidate_reads_evidence=True,
        update_evidence=False,
        aux_loss="margin_improvement",
        aux_weight=0.10,
        aux_margin=0.05,
        energy_source="candidate_evidence_compatibility",
    )

    add(
        "e_compat_ce_t2_w0p10",
        "E_evidence_clustering",
        "Auxiliary supervised candidate/evidence compatibility loss.",
        2,
        shared_update=True,
        gated_residual=True,
        norm_position="pre",
        candidate_reads_evidence=True,
        update_evidence=False,
        aux_loss="supervised_contrastive",
        aux_weight=0.10,
        energy_source="candidate_evidence_compatibility",
    )
    add(
        "e_triplet_t2_w0p10",
        "E_evidence_clustering",
        "Auxiliary triplet-style candidate/evidence compatibility margin loss.",
        2,
        shared_update=True,
        gated_residual=True,
        norm_position="pre",
        candidate_reads_evidence=True,
        update_evidence=False,
        aux_loss="triplet_margin",
        aux_weight=0.10,
        aux_margin=0.20,
        energy_source="candidate_evidence_compatibility",
    )
    add(
        "e_no_aux_compat_t2",
        "E_evidence_clustering",
        "Compatibility-energy transport without auxiliary contrastive loss.",
        2,
        shared_update=True,
        gated_residual=True,
        norm_position="pre",
        candidate_reads_evidence=True,
        update_evidence=False,
        aux_loss="none",
        aux_weight=0.0,
        energy_source="candidate_evidence_compatibility",
    )

    add(
        "f_stage5_plus_one_refinement",
        "F_minimal_control",
        "Stage 5-like direct readout plus one candidate refinement step.",
        1,
        shared_update=True,
        gated_residual=False,
        norm_position="pre",
        candidate_reads_evidence=True,
        update_evidence=False,
    )
    add(
        "f_stage5_candidate_residual_gating",
        "F_minimal_control",
        "Stage 5-like direct readout with one gated candidate residual refinement step.",
        1,
        shared_update=True,
        gated_residual=True,
        norm_position="pre",
        candidate_reads_evidence=True,
        update_evidence=False,
    )
    add(
        "f_stage5_workspace_only_m8",
        "F_minimal_control",
        "Stage 5-like direct readout with workspace slots and no dynamic evidence update.",
        1,
        workspace_slots=8,
        workspace_init="learned",
        evidence_writes_workspace=True,
        candidate_reads_workspace=True,
        candidate_conditioned_workspace_read=True,
        candidate_reads_evidence=False,
        update_evidence=False,
    )
    add(
        "f_stage5_energy_only_t1_w0p10",
        "F_minimal_control",
        "Stage 5-like one-step refinement with energy trajectory loss only.",
        1,
        shared_update=True,
        gated_residual=False,
        norm_position="pre",
        candidate_reads_evidence=True,
        update_evidence=False,
        aux_loss="margin_improvement",
        aux_weight=0.10,
    )
    add(
        "f_stage5_widened_param_control",
        "F_minimal_control",
        "Same protocol with transport disabled and a widened candidate-token-direct scorer as an extra-parameter control.",
        0,
        model_dim_multiplier=1.5,
        transport_disabled=True,
        candidate_reads_evidence=False,
        update_evidence=False,
    )

    return variants[: int(max_variants)] if int(max_variants) > 0 else variants


def stage7_ablation_plan(selected: Stage7VariantSpec) -> List[Stage7VariantSpec]:
    base = selected.transport_config

    def clone(name: str, description: str, updates: Dict[str, object]) -> Stage7VariantSpec:
        config = {**base, **updates}
        steps = int(config.get("refinement_steps", selected.refinement_steps))
        slots = int(config.get("workspace_slots", selected.workspace_slots))
        return Stage7VariantSpec(
            name=name,
            family=f"ablation_of_{selected.name}",
            description=description,
            transport_config=config,
            refinement_steps=steps,
            workspace_slots=slots,
            lr=selected.lr,
            epochs=selected.epochs,
            patience=selected.patience,
            gradient_clip_norm=selected.gradient_clip_norm,
            weight_decay=selected.weight_decay,
            dropout=selected.dropout,
            model_dim_multiplier=selected.model_dim_multiplier,
        )

    return [
        clone(f"{selected.name}_ablate_t0_disabled", "Transport disabled; Stage 5-like direct readout.", {"refinement_steps": 0, "transport_disabled": True}),
        clone(f"{selected.name}_ablate_t1", "One refinement step.", {"refinement_steps": 1, "transport_disabled": False}),
        clone(f"{selected.name}_ablate_t2", "Two refinement steps.", {"refinement_steps": 2, "transport_disabled": False}),
        clone(f"{selected.name}_ablate_t4", "Four refinement steps.", {"refinement_steps": 4, "transport_disabled": False}),
        clone(f"{selected.name}_ablate_no_aux", "Auxiliary transport loss removed.", {"aux_loss": "none", "aux_weight": 0.0}),
        clone(f"{selected.name}_ablate_no_evidence_update", "Evidence update removed.", {"update_evidence": False}),
        clone(f"{selected.name}_ablate_no_workspace", "Workspace slots removed.", {"workspace_slots": 0}),
        clone(f"{selected.name}_ablate_frozen_transport", "Transport update blocks frozen.", {"freeze_transport": True}),
        clone(f"{selected.name}_ablate_random_transport", "Random fixed transport update blocks.", {"random_transport": True, "freeze_transport": True}),
    ]


def stage7b_medium_variant_plan() -> List[Stage7VariantSpec]:
    base = {variant.name: variant for variant in stage7_variant_plan()}
    variants = [
        base["f_stage5_widened_param_control"],
        base["e_compat_ce_t2_w0p10"],
        base["f_stage5_workspace_only_m8"],
    ]
    compat = base["e_compat_ce_t2_w0p10"]
    compat_config = dict(compat.transport_config)

    def compat_clone(name: str, description: str, updates: Dict[str, object]) -> Stage7VariantSpec:
        config = {**compat_config, **updates}
        steps = int(config.get("refinement_steps", compat.refinement_steps))
        return Stage7VariantSpec(
            name=name,
            family="B7_compatibility_ablation",
            description=description,
            transport_config=config,
            refinement_steps=steps,
            workspace_slots=int(config.get("workspace_slots", 0)),
            lr=compat.lr,
            epochs=compat.epochs,
            patience=compat.patience,
            gradient_clip_norm=compat.gradient_clip_norm,
            weight_decay=compat.weight_decay,
            dropout=compat.dropout,
            model_dim_multiplier=compat.model_dim_multiplier,
        )

    variants.extend(
        [
            compat_clone(
                "b7_e_compat_ce_t2_no_aux",
                "Compatibility transport with the supervised compatibility auxiliary loss removed.",
                {"aux_loss": "none", "aux_weight": 0.0},
            ),
            compat_clone(
                "b7_e_compat_ce_t2_w0p05",
                "Compatibility transport with supervised compatibility auxiliary weight 0.05.",
                {"aux_loss": "supervised_contrastive", "aux_weight": 0.05},
            ),
            compat_clone(
                "b7_e_compat_ce_t2_w0p20",
                "Compatibility transport with supervised compatibility auxiliary weight 0.20.",
                {"aux_loss": "supervised_contrastive", "aux_weight": 0.20},
            ),
            compat_clone(
                "b7_e_compat_ce_t1_w0p10",
                "One-step compatibility transport with supervised compatibility auxiliary weight 0.10.",
                {"refinement_steps": 1, "aux_loss": "supervised_contrastive", "aux_weight": 0.10},
            ),
            compat_clone(
                "b7_e_compat_ce_t4_w0p10",
                "Four-step compatibility transport with supervised compatibility auxiliary weight 0.10.",
                {"refinement_steps": 4, "aux_loss": "supervised_contrastive", "aux_weight": 0.10},
            ),
        ]
    )

    workspace = base["f_stage5_workspace_only_m8"]
    workspace_config = dict(workspace.transport_config)

    def workspace_clone(name: str, description: str, updates: Dict[str, object], model_dim_multiplier: float | None = None) -> Stage7VariantSpec:
        config = {**workspace_config, **updates}
        return Stage7VariantSpec(
            name=name,
            family="B7_workspace_ablation",
            description=description,
            transport_config=config,
            refinement_steps=int(config.get("refinement_steps", workspace.refinement_steps)),
            workspace_slots=int(config.get("workspace_slots", workspace.workspace_slots)),
            lr=workspace.lr,
            epochs=workspace.epochs,
            patience=workspace.patience,
            gradient_clip_norm=workspace.gradient_clip_norm,
            weight_decay=workspace.weight_decay,
            dropout=workspace.dropout,
            model_dim_multiplier=workspace.model_dim_multiplier if model_dim_multiplier is None else float(model_dim_multiplier),
        )

    variants.extend(
        [
            workspace_clone(
                "b7_workspace_only_m4",
                "Workspace-only minimal control with four learned workspace slots.",
                {"workspace_slots": 4, "workspace_init": "learned"},
            ),
            workspace_clone(
                "b7_workspace_only_m16",
                "Workspace-only minimal control with sixteen learned workspace slots.",
                {"workspace_slots": 16, "workspace_init": "learned"},
            ),
            workspace_clone(
                "b7_workspace_only_m8_evidencepooled",
                "Workspace-only minimal control with eight evidence-pooled workspace slots.",
                {"workspace_slots": 8, "workspace_init": "evidence_pooled"},
            ),
            workspace_clone(
                "b7_workspace_disabled_param_control",
                "Workspace disabled with widened no-transport scoring as the feasible same-family capacity control.",
                {"workspace_slots": 0, "transport_disabled": True, "candidate_reads_workspace": False, "candidate_reads_evidence": False},
                model_dim_multiplier=1.5,
            ),
        ]
    )
    return variants


def stage7_variant_by_name(name: str, variants: Iterable[Stage7VariantSpec] | None = None) -> Stage7VariantSpec:
    for variant in variants or [*stage7_variant_plan(), *stage7b_medium_variant_plan()]:
        if variant.name == name:
            return variant
    raise KeyError(f"unknown Stage 7 variant: {name}")
