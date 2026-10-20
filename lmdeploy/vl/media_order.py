# Copyright (c) OpenMMLab. All rights reserved.
"""Track media references through the output of a HuggingFace chat template."""
import copy
import inspect
import re
import uuid
from functools import lru_cache

from jinja2 import nodes
from jinja2.visitor import NodeTransformer
from transformers.utils.chat_template_utils import _compile_jinja_template, get_json_schema


class MultimodalPrompt(str):
    """A rendered prompt and the source indices of its media, in output order.

    This remains a string for processors/tokenizers. The indices are consumed before feature extraction, never
    serialized into the model's prompt.
    """

    def __new__(cls, prompt: str, media_order: tuple[int, ...], media_contents: tuple[str, ...]):
        value = super().__new__(cls, prompt)
        value.media_order = media_order
        value.media_contents = media_contents
        return value

    def __getnewargs__(self):
        return str(self), self.media_order, self.media_contents


class _TrackMediaLoops(NodeTransformer):
    """Emit references around media loop bodies, including inside macros.

    Instrument the parsed AST rather than source text: Jinja's whitespace
    controls and the scope of assignments in the loop remain unchanged.
    """

    def visit_For(self, node, *args, **kwargs):
        self.generic_visit(node, *args, **kwargs)
        target = copy.deepcopy(node.target)
        target.set_ctx('load')

        def marker(end):
            return nodes.Output([
                nodes.Call(nodes.Name('_lmdeploy_media_marker', 'load'),
                           [copy.deepcopy(target), nodes.Const(end)], [], None, None)
            ]).set_lineno(node.lineno)

        node.body = [marker(False), *node.body, marker(True)]
        return node


@lru_cache(maxsize=128)
def _tracked_template(source: str):
    # Reuse Transformers' sandbox, filters and extensions (including generation
    # and loopcontrols), without patching its global compiler/cache.
    environment = _compile_jinja_template(source).environment
    tree = _TrackMediaLoops().visit(environment.parse(source))
    return environment.from_string(tree)


def render_with_media_order(tokenizer, messages: list[dict], **kwargs) -> MultimodalPrompt:
    """Render without changing messages, and recover media order from output.

    Supported templates render content blocks in loops, possibly through macros and captured strings. An
    omitted/duplicated reference, unsupported traversal, or a change to the ordinary prompt is rejected instead of
    guessing an order.
    """
    references = {}
    render_messages = []
    for message in messages:
        content = message.get('content')
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get('type') in ('image', 'video', 'audio') and 'data' in part:
                    part = dict(part)
                    references[id(part)] = len(references)
                parts.append(part)
            message = dict(message, content=parts)
        render_messages.append(message)

    # Keep tokenizer validation, tool conversion and prompt behavior as the
    # reference. No instrumentation reaches the tokenizer or inference engine.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    if not references:
        return MultimodalPrompt(prompt, (), ())

    prefix = f'\x00lmdeploy_media_{uuid.uuid4().hex}:'

    def marker(value, end):
        if isinstance(value, tuple):
            return ''.join(marker(item, end) for item in value)
        index = references.get(id(value))
        return '' if index is None else f'{prefix}{index}:{int(end)}\x00'

    render_kwargs = {**tokenizer.special_tokens_map, **kwargs}
    source = tokenizer.get_chat_template(render_kwargs.pop('chat_template', None), render_kwargs.get('tools'))
    tools = render_kwargs.get('tools')
    if tools is not None:
        render_kwargs['tools'] = [
            get_json_schema(tool) if inspect.isfunction(tool) or inspect.ismethod(tool) else tool for tool in tools
        ]
    # These options transform the final output in Transformers; they are not
    # supported by this internal string-only rendering path.
    if render_kwargs.get('continue_final_message') or render_kwargs.get('return_assistant_tokens_mask'):
        raise ValueError('Media reference tracking does not support continuation or assistant-token masks.')
    render_kwargs.pop('tokenizer_kwargs', None)
    render_kwargs.setdefault('documents', None)
    render_kwargs.setdefault('tools', None)
    render_kwargs['_lmdeploy_media_marker'] = marker
    tracked = _tracked_template(source).render(messages=render_messages, **render_kwargs)
    pattern = re.compile(re.escape(prefix) + r'(\d+):([01])\x00')
    order = []
    contents = []
    active = None
    for match in pattern.finditer(tracked):
        index, end = int(match[1]), int(match[2])
        if not end:
            if active is not None:
                raise ValueError('Chat template nests media references; media order cannot be determined.')
            active = (index, match.end())
        else:
            if active is None or active[0] != index:
                raise ValueError('Chat template emitted an incomplete media reference.')
            # Templates may inspect media in preliminary loops without emitting
            # anything. Only references accompanying output determine order.
            if tracked[active[1]:match.start()].strip():
                order.append(index)
                contents.append(tracked[active[1]:match.start()])
            active = None
    if active is not None or sorted(order) != list(range(len(references))):
        raise ValueError('Chat template must render each media block exactly once in a content loop; '
                         'media reference order cannot be determined.')
    if pattern.sub('', tracked) != prompt:
        raise ValueError('Tracking media references changed the rendered chat template; refusing to guess media order.')
    return MultimodalPrompt(prompt, tuple(order), tuple(contents))
