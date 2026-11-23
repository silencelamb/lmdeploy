# Copyright (c) OpenMMLab. All rights reserved.
"""The retained decode rows sharing a target forward with admitted prompt rows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from lmdeploy.pytorch.model_inputs import ModelInputsDelta

    from .cache_inputs import CacheCheckpointInputs
    from .logits_process import SamplingInputs


@dataclass
class MixedDecodeInputs:
    delta: ModelInputsDelta
    sampling_inputs: SamplingInputs
    cache_inputs: CacheCheckpointInputs | None

    def to_device(self, device, non_blocking=False):
        return MixedDecodeInputs(self.delta.to_device(device, non_blocking),
                                 self.sampling_inputs.to_device(device, non_blocking),
                                 self.cache_inputs.to_device(device, non_blocking) if self.cache_inputs else None)

    def record_stream(self, stream: torch.cuda.Stream):
        self.delta.record_stream(stream)
        self.sampling_inputs.record_stream(stream)
        if self.cache_inputs is not None:
            self.cache_inputs.record_stream(stream)
