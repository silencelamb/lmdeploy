# Copyright (c) OpenMMLab. All rights reserved.
"""Desired compatibility behavior; strict xfails document prototype blockers.

Use --runxfail to reproduce the failures instead of treating them as expected.
"""
import pytest
from PIL import Image

from lmdeploy.vl.model.base import VisionModel


@pytest.mark.parametrize('source', [
    pytest.param("{% if messages[0].content[0].type == 'image' %}<|image|>{% endif %}", id='direct-index'),
    pytest.param('{% for i in range(messages[0].content|length) %}{% set p = messages[0].content[i] %}'
                 "{% if p.type == 'image' %}<|image|>{% endif %}{% endfor %}", id='index-loop'),
    pytest.param('{% macro emit(content) %}{% for p in content %}  '
                 "{% if p.type == 'image' %}<|image|>{% endif %}  {% endfor %}{% endmacro %}"
                 '{{ emit(messages[0].content).strip() }}', id='macro-strip'),
    pytest.param("{% for p in messages[0].content %}{% if p.type == 'image' %}<|image|>"
                 '{% break %}{% endif %}{% endfor %}', id='break'),
    pytest.param("{% for p in messages[0].content %}{% if p.type == 'image' %}<|image|>"
                 '{% continue %}{% endif %}{% endfor %}', id='continue'),
])
@pytest.mark.xfail(strict=True, raises=ValueError, reason='Loop markers do not cover all valid Jinja output paths.')
def test_valid_custom_templates_should_remain_usable(hf_media_template, source):
    messages = [dict(role='user', content=[dict(type='image', data=Image.new('RGB', (28, 28)))])]
    template = hf_media_template(source=source)
    ordinary = template.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert ordinary == '<|image|>'
    prompt = VisionModel.get_input_prompt(messages, template)
    assert prompt == ordinary
    assert prompt.media_order == (0,)


@pytest.mark.parametrize('modality,token', [
    ('image', '<|image_pad|>'), ('video', '<|video_pad|>'), ('audio', '<|audio_pad|>'),
])
@pytest.mark.xfail(strict=True, raises=ValueError, reason='Qwen3-Omni emits accumulated strings outside the loop.')
def test_official_omni_accumulation_should_remain_usable(hf_media_template, modality, token):
    # Template-only probe: no audio/video loading or inference is involved.
    messages = [dict(role='user', content=[dict(type=modality, data='parsed-media')])]
    template = hf_media_template('qwen3_omni')
    ordinary = template.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert ordinary.count(token) == 1
    prompt = VisionModel.get_input_prompt(messages, template)
    assert prompt == ordinary
    assert prompt.media_order == (0,)
