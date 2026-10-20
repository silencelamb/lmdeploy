# Copyright (c) OpenMMLab. All rights reserved.
import copy
import pickle
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from lmdeploy.vl.model.base import MultimodalSpecialTokens, VisionModel
from lmdeploy.vl.model.glm5_next import GLM5NextVisionModel
from lmdeploy.vl.model.internvl import InternVLVisionModel

TOKENS = MultimodalSpecialTokens(image_token='<image>', video_token='<video>', audio_token='<audio>')
BODY = """{% for part in message.content %}
{% if part.type == 'text' %}{{ part.text }}{% else %}{{ '<' + part.type + '>' }}{% endif %}
{% endfor %}"""


def _messages():
    return [dict(role='user', content=[
        dict(type='text', text='before'),
        dict(type='image', data=Image.new('RGB', (2, 2), color)),
        dict(type='text', text='after'),
    ]) for color in ('red', 'blue', 'lime')]


@pytest.mark.parametrize('reverse', [False, True])
def test_template_controls_media_order_and_preserves_prompt(hf_media_template, reverse):
    source = '{% for message in messages' + (' | reverse' if reverse else '') + ' %}' + BODY + '{% endfor %}'
    template = hf_media_template(source=source)
    messages = _messages()
    prompt = VisionModel.get_input_prompt(messages, template)
    assert prompt == template.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    items = VisionModel.collect_multimodal_items(messages, prompt, TOKENS)
    expected = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
    assert [image.getpixel((0, 0)) for _, image, _ in items] == (expected[::-1] if reverse else expected)
    assert '\x00' not in prompt
    assert copy.deepcopy(prompt).media_order == prompt.media_order
    assert pickle.loads(pickle.dumps(prompt)).media_contents == prompt.media_contents


def test_captured_macros_use_output_order_not_execution_order(hf_media_template):
    source = ('{% macro content(message) %}' + BODY + '{% endmacro %}'
              '{% set first = content(messages[0]) %}'
              '{% set second = content(messages[1]) %}'
              '{% set third = content(messages[2]) %}'
              '{{ third + first + second }}')
    template = hf_media_template(source=source)
    prompt = VisionModel.get_input_prompt(_messages(), template)
    assert prompt.media_order == (2, 0, 1)


def test_preliminary_scan_and_loop_assignments(hf_media_template):
    source = """{% set ns = namespace(count=0) %}
{% for message in messages %}{% for part in message.content %}
{% if part.type == 'image' %}{% set ns.count = ns.count + 1 %}{% endif %}
{% endfor %}{% endfor %}
{{ ns.count }}
{% for message in messages | reverse %}""" + BODY + '{% endfor %}'
    prompt = VisionModel.get_input_prompt(_messages(), hf_media_template(source=source))
    assert prompt.media_order == (2, 1, 0)
    assert prompt.lstrip().startswith('3')


@pytest.mark.parametrize('source', [
    # Index-based rendering cannot be attributed to a content-loop reference.
    '{{ "<image><image><image>" }}',
    # Dropping or duplicating media must not silently keep the input order.
    '{% for message in messages[:2] %}' + BODY + '{% endfor %}',
    '{% for message in messages + messages %}' + BODY + '{% endfor %}',
])
def test_untraceable_or_incomplete_templates_are_rejected(hf_media_template, source):
    with pytest.raises(ValueError, match='media block exactly once'):
        VisionModel.get_input_prompt(_messages(), hf_media_template(source=source))


def test_reference_must_enclose_the_actual_placeholder(hf_media_template):
    # Merely visiting a block and emitting its label does not locate its image.
    source = ('{% for message in messages %}{% for part in message.content %}'
              '{% if part.type == "image" %}label{% endif %}{% endfor %}{% endfor %}'
              '<image><image><image>')
    messages = _messages()
    prompt = VisionModel.get_input_prompt(messages, hf_media_template(source=source))
    with pytest.raises(ValueError, match='exactly one media placeholder'):
        VisionModel.collect_multimodal_items(messages, prompt, TOKENS)


def test_untracked_extra_placeholder_is_rejected(hf_media_template):
    source = '{% for message in messages %}' + BODY + '{% endfor %}<image>'
    messages = _messages()
    prompt = VisionModel.get_input_prompt(messages, hf_media_template(source=source))
    with pytest.raises(ValueError, match='outside their tracked references'):
        VisionModel.collect_multimodal_items(messages, prompt, TOKENS)


def test_prompt_changes_from_output_filters_are_rejected(hf_media_template):
    source = ('{% macro content(message) %}  ' + BODY + '  {% endmacro %}'
              '{% for message in messages %}{{ content(message) | upper }}{% endfor %}')
    with pytest.raises(ValueError, match='media block exactly once|changed the rendered'):
        VisionModel.get_input_prompt(_messages(), hf_media_template(source=source))


def test_mixed_media_references_are_collected_in_render_order(hf_media_template):
    messages = [dict(role='user', content=[
        dict(type='image', data=object()), dict(type='video', data=object()), dict(type='audio', data=object()),
    ])]
    body = BODY.replace('message.content', 'message.content | reverse')
    source = '{% for message in messages %}' + body + '{% endfor %}'
    prompt = VisionModel.get_input_prompt(messages, hf_media_template(source=source))
    items = VisionModel.collect_multimodal_items(messages, prompt, TOKENS)
    assert [modality for modality, _, _ in items] == ['audio', 'video', 'image']
    assert [data for _, data, _ in items] == [part['data'] for part in reversed(messages[0]['content'])]


def test_concurrent_renders_do_not_share_request_state(hf_media_template):
    template = hf_media_template(source='{% for message in messages | reverse %}' + BODY + '{% endfor %}')

    def render(size):
        messages = _messages()[:size]
        prompt = VisionModel.get_input_prompt(messages, template)
        assert prompt.media_order == tuple(reversed(range(size)))
        assert '\x00' not in prompt
        return len(VisionModel.collect_multimodal_items(messages, prompt, TOKENS))

    sizes = [1, 2, 3] * 8
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(render, sizes)) == sizes


def test_production_preprocessor_receives_images_in_rendered_order(hf_media_template):
    template = hf_media_template(source='{% for message in messages | reverse %}' + BODY + '{% endfor %}')
    messages = _messages()
    prompt = VisionModel.get_input_prompt(messages, template)
    model = GLM5NextVisionModel.__new__(GLM5NextVisionModel)
    model._shared_video_token = False
    model.mm_tokens = TOKENS
    model.processor = Mock(return_value={'input_ids': torch.tensor([[1, 2]])})
    model.preprocess(messages, prompt)
    kwargs = model.processor.call_args.kwargs
    assert [image.getpixel((0, 0)) for image in kwargs['images']] == [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    assert kwargs['text'] == [str(prompt)]
    assert '\x00' not in kwargs['text'][0]


@pytest.mark.parametrize('name', ['internvl3', 'internvl3_5'])
@pytest.mark.parametrize('legacy', [(), ('user',), ('tool',), ('user', 'tool')])
def test_internvl_tool_images_keep_roles_ids_and_placeholders(hf_media_template, name, legacy):
    template = hf_media_template(name)
    messages = _messages()[:2]
    messages[1].update(role='tool', tool_call_id='screenshot', name='capture')
    for message in messages:
        if message['role'] in legacy:
            message['content'][0]['text'] += '<IMAGE_TOKEN>'
    original = copy.deepcopy(messages)
    seen = []
    render = template.messages2prompt

    def capture(prompt_messages, **kwargs):
        seen.extend(prompt_messages)
        return render(prompt_messages, **kwargs)

    template.messages2prompt = capture
    model = InternVLVisionModel.__new__(InternVLVisionModel)
    model.image_token = '<IMG_CONTEXT>'
    model.image_token_id = 13
    prompt, token = model.proc_messages(messages, template)
    assert prompt.count(token) == len(VisionModel.collect_multimodal_items(messages)) == 2
    assert seen[1]['role'] == 'tool'
    assert seen[1]['tool_call_id'] == 'screenshot'
    assert seen[1]['name'] == 'capture'
    assert messages == original
    preprocessed = messages + [dict(role='preprocess', content=[
        dict(image_token_id=13, image_tokens=2) for _ in range(2)
    ])]
    result = model.to_pytorch_aux(preprocessed, prompt, token, SimpleNamespace(encode=lambda text, **kw: [1]))
    assert len(result['multimodal']) == 2
