from __future__ import annotations

from typing import Dict, Iterable, List

import torch


class ForwardHookRecorder:
    """Small generic forward-hook recorder for transformer block activations."""

    def __init__(self, modules: Iterable[torch.nn.Module]) -> None:
        self.modules = list(modules)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.outputs: Dict[int, torch.Tensor] = {}

    def __enter__(self) -> "ForwardHookRecorder":
        for index, module in enumerate(self.modules):
            self.handles.append(module.register_forward_hook(self._make_hook(index)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, index: int):  # type: ignore[no-untyped-def]
        def hook(_module, _inputs, output):  # type: ignore[no-untyped-def]
            tensor = output[0] if isinstance(output, tuple) else output
            self.outputs[index] = tensor.detach()

        return hook

    def stacked_final_token(self) -> torch.Tensor:
        if not self.outputs:
            raise RuntimeError("no hook outputs captured")
        ordered = [self.outputs[i][:, -1, :] for i in sorted(self.outputs)]
        return torch.stack(ordered, dim=1)

