```python
"""Supplement the separate draft forward entry; never modify primary-run hooks."""
import functools

import warm_observer as observed


class DraftShapeWorker(observed.WarmWorker):
    def build_model(self):
        result=super().build_model()
        from lmdeploy.pytorch.spec_decode.proposers import base
        original=base.draft_model_forward
        @functools.wraps(original)
        def forward(model,inputs,*args,**kwargs):
            signature=('draft:'+type(getattr(model,'model',model)).__name__,bool(inputs.is_decoding),
                       inputs.input_ids.numel(),inputs.seq_length.numel())
            observed.COUNTS[(observed.phase(),signature)]+=1
            if signature not in observed.SEEN:
                observed.SEEN.add(signature)
                observed.event('new_forward_shape',signature=signature,role='draft')
            return original(model,inputs,*args,**kwargs)
        base.draft_model_forward=forward
        observed.event('draft_hook_installed',entry='spec_decode.proposers.base.draft_model_forward')
        return result
```
