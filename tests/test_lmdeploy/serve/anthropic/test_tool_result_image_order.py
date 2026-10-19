"""Regression: tool ids, image ownership, and preprocessing order stay aligned."""
import copy

import pytest

from lmdeploy.serve.anthropic.adapter import to_openai_messages
from lmdeploy.serve.anthropic.protocol import MessagesRequest


def image(label):
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': label}}


def text(value):
    return {'type': 'text', 'text': value}


def request(result_ids=('b', 'a'), call_ids=('a', 'b')):
    return MessagesRequest.model_validate({
        'model': 'test', 'max_tokens': 32,
        'messages': [
            {'role': 'user', 'content': 'Compare pages.'},
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': identifier, 'name': 'screenshot', 'input': {'page': identifier}}
                for identifier in call_ids]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': identifier,
                 'content': [text(identifier + '-before'), image(identifier + '-1'),
                             text(identifier + '-between'), image(identifier + '-2')]}
                for identifier in result_ids]},
        ]})


def images(messages):
    return [part['image_url']['url'].split(',')[-1]
            for message in messages if isinstance(message.get('content'), list)
            for part in message['content'] if part.get('type') == 'image_url']


def test_reverse_results_keep_multiple_images_with_their_tool_ids():
    original = request()
    before = copy.deepcopy(original.model_dump())
    converted = to_openai_messages(original)
    assert [m['role'] for m in converted] == ['user', 'assistant', 'tool', 'tool']
    assert [m['tool_call_id'] for m in converted[2:]] == ['a', 'b']
    assert images(converted) == ['a-1', 'a-2', 'b-1', 'b-2']
    assert [p['type'] for p in converted[2]['content']] == ['text', 'image_url', 'text', 'image_url']
    assert converted[2]['content'][2]['text'] == 'a-between'
    assert original.model_dump() == before


def test_forward_and_reverse_results_have_identical_preprocessor_input():
    assert to_openai_messages(request(('a', 'b'))) == to_openai_messages(request(('b', 'a')))


def test_order_follows_calls_not_alphabetical_ids():
    converted = to_openai_messages(request(('a', 'b'), ('b', 'a')))
    assert [m['tool_call_id'] for m in converted[2:]] == ['b', 'a']
    assert images(converted) == ['b-1', 'b-2', 'a-1', 'a-2']


@pytest.mark.parametrize('result_ids,call_ids', [
    (('b', 'unknown'), ('a', 'b')),
    (('b', 'b'), ('a', 'b')),
    (('b', 'a'), ('a', 'a')),
])
def test_ambiguous_ids_are_not_guessed_or_dropped(result_ids, call_ids):
    converted = to_openai_messages(request(result_ids, call_ids))
    assert [m['tool_call_id'] for m in converted[2:]] == list(result_ids)
    assert len(images(converted)) == 4


def test_rounds_are_not_merged_or_cross_associated():
    original = request()
    second = request(('d', 'c'), ('c', 'd'))
    original.messages.extend(second.messages)
    converted = to_openai_messages(original)
    assert [m['tool_call_id'] for m in converted if m['role'] == 'tool'] == ['a', 'b', 'c', 'd']
    assert images(converted) == ['a-1', 'a-2', 'b-1', 'b-2', 'c-1', 'c-2', 'd-1', 'd-2']


def test_plain_text_results_keep_existing_format_and_order():
    original = request()
    for block in original.messages[-1].content:
        block.content = [text(block.tool_use_id + '-one'), text(block.tool_use_id + '-two')]
    converted = to_openai_messages(original)
    assert [(m['tool_call_id'], m['content']) for m in converted[2:]] == [('b', 'b-one\nb-two'), ('a', 'a-one\na-two')]


def test_mixed_text_and_image_tools_reorder_as_whole_results():
    original = request()
    original.messages[-1].content[0].content = 'beta text only'
    converted = to_openai_messages(original)
    assert [m['tool_call_id'] for m in converted[2:]] == ['a', 'b']
    assert converted[3]['content'] == 'beta text only'
    assert images(converted) == ['a-1', 'a-2']
