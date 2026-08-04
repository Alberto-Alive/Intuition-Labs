from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class GateSpec:
    module_name: str
    groups: int


class ActivationGateController:
    """
    Generic activation-group gating for PyTorch modules.

    It attaches forward hooks to named modules and multiplies output channels by
    a binary group mask. This is useful for intervention-based strategy
    discovery. It does not by itself rewrite the network into a physically
    smaller graph; exporting a structurally pruned model is a separate step.
    """

    def __init__(
        self,
        model: nn.Module,
        specs: list[GateSpec],
    ):
        self.model = model
        self.specs = specs
        self._mask = np.ones(sum(s.groups for s in specs), dtype=np.int8)
        self._handles = []

        modules = dict(model.named_modules())
        offset = 0

        for spec in specs:
            if spec.module_name not in modules:
                raise KeyError(f"Unknown module: {spec.module_name}")

            module = modules[spec.module_name]
            start = offset
            end = offset + spec.groups
            offset = end

            def make_hook(start=start, end=end, groups=spec.groups):
                def hook(_module, _inputs, output):
                    if not torch.is_tensor(output):
                        return output

                    channels = output.shape[1] if output.ndim >= 2 else output.shape[-1]
                    if channels % groups != 0:
                        raise ValueError(
                            f"Channel count {channels} is not divisible by groups={groups}"
                        )

                    group_width = channels // groups
                    group_mask = torch.tensor(
                        self._mask[start:end],
                        dtype=output.dtype,
                        device=output.device,
                    )
                    channel_mask = group_mask.repeat_interleave(group_width)

                    shape = [1] * output.ndim
                    if output.ndim >= 2:
                        shape[1] = channels
                    else:
                        shape[-1] = channels

                    return output * channel_mask.reshape(shape)

                return hook

            self._handles.append(module.register_forward_hook(make_hook()))

    @property
    def n_modules(self) -> int:
        return len(self._mask)

    def set_mask(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=np.int8)
        if mask.shape != self._mask.shape:
            raise ValueError(
                f"Expected mask shape {self._mask.shape}, got {mask.shape}"
            )
        self._mask = mask.copy()

    def clear(self) -> None:
        self._mask[:] = 1

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
