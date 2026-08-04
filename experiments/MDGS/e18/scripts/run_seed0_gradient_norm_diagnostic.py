"""Run the approved seed-0 E16 vs E18 gradient norm safety diagnostic."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e18.extrapolation.experiment import run_seed0_gradient_norm_diagnostic


def main() -> None:
    summaries = run_seed0_gradient_norm_diagnostic()
    payload = [
        {
            "step": summary.step,
            "e16_gradient_norm": summary.e16_gradient_norm,
            "e18_gradient_norm": summary.e18_gradient_norm,
            "ratio_e18_over_e16": summary.ratio_e18_over_e16,
            "layer1_modulated_fraction": summary.layer1_modulated_fraction,
            "layer2_modulated_fraction": summary.layer2_modulated_fraction,
        }
        for summary in summaries
    ]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
