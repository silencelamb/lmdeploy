# Copyright (c) OpenMMLab. All rights reserved.
"""Version 1 of the PyTorch engine's model-specialized decode interface."""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch.distributed as dist

if TYPE_CHECKING:
    import torch

    from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
    from lmdeploy.pytorch.distributed import DistContext
    from lmdeploy.pytorch.model_inputs import ModelInputs, StepContext


@dataclass(frozen=True)
class DecodePlan:
    """One exact, unpadded, single-query batch per participating rank.

    max_kv_len bounds every row including the current token. None accepts the entire engine-supported range. Plans are
    tried in declaration order.
    """

    name: str
    batch_size: int
    max_kv_len: int | None = None
    graph_safe: bool = False


@dataclass(frozen=True)
class BuildContext:
    """Borrowed deployment objects.

    Model parameters are in backend layout.
    """

    model: torch.nn.Module
    model_config: ModelConfig
    cache_config: CacheConfig
    backend_config: BackendConfig
    dist: DistContext
    device: torch.device


class SpecializedExecutor(ABC):
    """One instance per target worker and weight generation."""

    api_version: ClassVar[int]
    artifact_id: ClassVar[str]

    @classmethod
    @abstractmethod
    def describe(cls, ctx: BuildContext) -> tuple[DecodePlan, ...]:
        """Inspect static support without allocating, mutating or
        communicating."""

    def __init__(self, ctx: BuildContext):
        """Prepare derived weights and workspace before KV capacity is
        measured."""
        self.model = ctx.model

    @abstractmethod
    def run(self, plan: DecodePlan, **model_inputs) -> Any:
        """Return the model's forward output and perform the same state
        writes."""

    def close(self) -> None:
        """Release provider resources after all users and graphs are
        retired."""


@dataclass
class SpecializedExecution:
    plans: tuple[DecodePlan, ...]
    executor: SpecializedExecutor
    dp_size: int

    def propose(self, inputs: ModelInputs | StepContext, batch_size: int) -> int:
        """Zero means standard execution.

        Use only host-resident metadata.
        """
        if (not inputs.is_decoding or inputs.is_dummy or inputs.is_chunk or inputs.enable_microbatch
                or inputs.vision_inputs is not None
                or inputs.max_q_seqlen != 1 or inputs.input_ids.numel() != batch_size
                or inputs.max_kv_seqlen is None or inputs.max_kv_seqlen < 1):
            return 0
        for index, plan in enumerate(self.plans, start=1):
            if plan.batch_size == batch_size and (plan.max_kv_len is None or inputs.max_kv_seqlen <= plan.max_kv_len):
                return index
        return 0

    def select(self, ctx: StepContext, batch_size: int) -> DecodePlan | None:
        index = ctx.specialized_plan
        if index is None:
            index = self.propose(ctx, batch_size) if self.dp_size == 1 else 0
        return self.plans[index - 1] if index else None


def agree_on_plan(proposals: list[int]) -> int:
    """Every rank must enter the same collective program, including misses."""
    first = proposals[0]
    return first if all(index == first for index in proposals) else 0


def _describe(path: str, ctx: BuildContext):
    module, sep, name = path.partition(':')
    if not sep or not module or not name:
        raise ValueError('specialized_executor must be a Python module:Class')
    provider = getattr(importlib.import_module(module), name)
    if (not isinstance(provider, type) or not issubclass(provider, SpecializedExecutor)
            or getattr(provider, 'api_version', None) != 1):
        raise ValueError(f'{path} must implement SpecializedExecutor API version 1')
    if not isinstance(getattr(provider, 'artifact_id', None), str) or not provider.artifact_id:
        raise ValueError(f'{path} must declare an artifact_id for its code and collective protocol')
    plans = tuple(provider.describe(ctx))
    if (any(not isinstance(p, DecodePlan) or not p.name or type(p.batch_size) is not int or p.batch_size < 1
            or (p.max_kv_len is not None and (type(p.max_kv_len) is not int or p.max_kv_len < 1))
            or type(p.graph_safe) is not bool for p in plans)
            or len({p.name for p in plans}) != len(plans)):
        raise ValueError(f'{path} published invalid decode plans: {plans}')
    return provider, plans


def _gather(ctx: BuildContext, value):
    size = ctx.dist.dist_config.world_size
    if size == 1:
        return [value]
    results = [None] * size
    dist.all_gather_object(results, value, group=ctx.dist.cpu_group)
    return results


def build_specialized_execution(path: str, ctx: BuildContext) -> SpecializedExecution | None:
    """Agree on coverage before preparation.

    Propagate rank-local startup errors.
    """
    error = None
    try:
        provider, plans = _describe(path, ctx)
    except Exception as exc:
        provider, plans, error = None, (), exc
    signature = (path, getattr(provider, 'artifact_id', None), plans, repr(error) if error else None)
    signatures = _gather(ctx, signature)
    if error or any(other != signature for other in signatures):
        raise RuntimeError(f'Invalid or inconsistent specialized executor on workers: {signatures}') from error
    if not plans:
        return None
    executor = None
    try:
        executor = provider(ctx)
    except Exception as exc:
        error = exc
    failures = _gather(ctx, repr(error) if error else None)
    if any(failures):
        if executor is not None:
            executor.close()
        raise RuntimeError(f'Specialized executor preparation failed on workers: {failures}') from error
    return SpecializedExecution(plans, executor, ctx.dist.dist_config.dp)
