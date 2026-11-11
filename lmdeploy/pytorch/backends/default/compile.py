# Copyright (c) OpenMMLab. All rights reserved.
"""Call-site compilation policy for shared vision helpers."""
from functools import wraps

import torch

from lmdeploy.pytorch.devices import get_device_manager


def compile_for_cuda(**options):
    def decorate(func):
        compiled = None
        @wraps(func)
        def call(*args, **kwargs):
            nonlocal compiled
            if get_device_manager().current_context().device_type != 'cuda':
                return func(*args, **kwargs)
            if compiled is None:
                compiled = torch.compile(func, **options)
            return compiled(*args, **kwargs)
        return call
    return decorate
