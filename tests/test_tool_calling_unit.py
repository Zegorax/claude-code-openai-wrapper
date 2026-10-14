"""Unit tests for OpenAI function calling support (src/tool_calling.py)."""

import json

import pytest
from pydantic import ValidationError

from src.models import ChatCompletionRequest, Message
from src.tool_calling import (
    ToolCallStreamParser,
    build_tools_section,
    parse_complete_reply,
    render_prompt,
    retry_prompt,
    to_openai_tool_calls,
)

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
}
LIST_DIR = {"type": "function", "function": {"name": "list_dir"}}


def make_request(messages=None, **kwargs):
    kwargs.setdefault("tools", [READ_FILE, LIST_DIR])
    return ChatCompletionRequest(messages=messages or [{"role": "user", "content": "hi"}], **kwargs)


def call_block(name, arguments=None):
    payload = {"name": name}
    if arguments is not None:
        payload["arguments"] = arguments
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


class TestSchema:
    def test_tool_role_message(self):
        msg = Message(role="tool", content="result", tool_call_id="call_1")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"

    def test_assistant_tool_calls_with_null_content(self):
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        )
        assert msg.content is None
        assert msg.tool_calls[0].function.name == "f"

    def test_developer_role(self):
        assert Message(role="developer", content="x").role == "developer"

    def test_image_url_part_becomes_placeholder(self):
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Look:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        )
        assert msg.content == "Look:\n[image omitted]"

    def test_request_tool_fields(self):
        req = make_request(
            tool_choice={"type": "function", "function": {"name": "read_file"}},
            parallel_tool_calls=False,
        )
        assert req.tools[0].function.name == "read_file"
        assert req.parallel_tool_calls is False
        assert req.tool_calling_enabled()

    def test_tool_choice_values(self):
        for choice in ("auto", "none", "required"):
            assert make_request(tool_choice=choice).tool_choice == choice
        with pytest.raises(ValidationError):
            make_request(tool_choice="sometimes")

    def test_tool_choice_none_disables_tools(self):
        assert not make_request(tool_choice="none").tool_calling_enabled()

    def test_uses_tool_features(self):
        assert make_request().uses_tool_features()
        plain = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        assert not plain.uses_tool_features()
        history = ChatCompletionRequest(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": "x", "tool_call_id": "c"},
            ]
        )
        assert history.uses_tool_features()

    def test_message_dump_omits_unset_tool_fields(self):
        dumped = Message(role="assistant", content="hi").model_dump()
        assert dumped == {"role": "assistant", "content": "hi", "name": None}


class TestRendering:
    def test_tools_section_only_with_tools(self):
        _, system = render_prompt(make_request())
        assert "<tools>" in system
        assert '"name": "read_file"' in system

        _, system = render_prompt(make_request(tools=None))
        assert system is None

        _, system = render_prompt(
            make_request(messages=[{"role": "system", "content": "Be brief."}], tool_choice="none")
        )
        assert system == "Be brief."

    def test_tools_section_is_deterministic(self):
        a = make_request()
        reordered = json.loads(json.dumps(READ_FILE))
        props = reordered["function"]["parameters"]
        reordered["function"]["parameters"] = dict(reversed(list(props.items())))
        b = make_request(tools=[reordered, LIST_DIR])
        assert build_tools_section(a) == build_tools_section(b)

    def test_tool_choice_instructions(self):
        assert "You must call at least one tool." in build_tools_section(
            make_request(tool_choice="required")
        )
        forced = make_request(tool_choice={"type": "function", "function": {"name": "list_dir"}})
        assert "You must call `list_dir`." in build_tools_section(forced)
        single = make_request(parallel_tool_calls=False)
        assert "Call at most one tool per reply." in build_tools_section(single)
        assert "at most one" not in build_tools_section(make_request())

    def test_multiple_system_messages_joined(self):
        req = make_request(
            messages=[
                {"role": "system", "content": "First."},
                {"role": "developer", "content": "Second."},
                {"role": "user", "content": "hi"},
            ],
            tools=None,
        )
        _, system = render_prompt(req)
        assert system == "First.\n\nSecond."

    def test_full_history(self):
        req = make_request(
            messages=[
                {"role": "user", "content": "Read a and b"},
                {
                    "role": "assistant",
                    "content": "Reading both.",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a"}'},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "b"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "AAA"},
                {"role": "tool", "tool_call_id": "call_b", "content": "BBB"},
            ]
        )
        prompt, _ = render_prompt(req)
        assert prompt == (
            "Human: Read a and b\n\n"
            "Assistant: Reading both.\n"
            '<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>\n'
            '<tool_call>{"name": "read_file", "arguments": {"path": "b"}}</tool_call>\n\n'
            'Human: <tool_result id="call_a" name="read_file">\nAAA\n</tool_result>\n'
            '<tool_result id="call_b" name="read_file">\nBBB\n</tool_result>'
        )

    def test_tool_call_only_assistant_message(self):
        req = make_request(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "list_dir"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "x"},
            ]
        )
        prompt, _ = render_prompt(req)
        assert 'Assistant: <tool_call>{"name": "list_dir", "arguments": {}}</tool_call>' in prompt
        assert '<tool_result id="c1" name="list_dir">' in prompt
        assert not prompt.endswith("Please continue.")

    def test_tool_result_name_falls_back_to_message_name(self):
        req = make_request(
            messages=[{"role": "tool", "tool_call_id": "zz", "name": "list_dir", "content": "x"}]
        )
        prompt, _ = render_prompt(req)
        assert '<tool_result id="zz" name="list_dir">' in prompt

    def test_trailing_assistant_gets_continue(self):
        req = make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        )
        prompt, _ = render_prompt(req)
        assert prompt.endswith("Human: Please continue.")

    def test_retry_prompt(self):
        prompt = retry_prompt("Human: hi", "<tool_call>{bad</tool_call>")
        assert prompt.startswith("Human: hi\n\nAssistant: <tool_call>{bad</tool_call>\n\nHuman: ")


class TestParsing:
    def test_single_call(self):
        content, parsed = parse_complete_reply(
            call_block("read_file", {"path": "a"}), make_request()
        )
        assert content == ""
        assert parsed.calls == [{"name": "read_file", "arguments": {"path": "a"}}]

    def test_multiple_calls(self):
        text = call_block("read_file", {"path": "a"}) + "\n" + call_block("list_dir", {})
        _, parsed = parse_complete_reply(text, make_request())
        assert [c["name"] for c in parsed.calls] == ["read_file", "list_dir"]

    def test_leading_text(self):
        text = "Let me check.\n\n" + call_block("list_dir", {})
        content, parsed = parse_complete_reply(text, make_request())
        assert content == "Let me check."
        assert len(parsed.calls) == 1

    def test_code_fence_inside_tags(self):
        text = '<tool_call>\n```json\n{"name": "list_dir", "arguments": {}}\n```\n</tool_call>'
        _, parsed = parse_complete_reply(text, make_request())
        assert parsed.calls == [{"name": "list_dir", "arguments": {}}]

    def test_code_fence_around_tags(self):
        text = "Calling.\n```xml\n" + call_block("list_dir", {}) + "\n```"
        content, parsed = parse_complete_reply(text, make_request())
        assert content == "Calling."
        assert len(parsed.calls) == 1

    def test_closing_fence_before_call_is_kept(self):
        text = "Code:\n```py\nx = 1\n```\n" + call_block("list_dir", {})
        content, _ = parse_complete_reply(text, make_request())
        assert content == "Code:\n```py\nx = 1\n```"

    def test_parameters_alias_and_missing_arguments(self):
        text = '<tool_call>{"name": "read_file", "parameters": {"path": "a"}}</tool_call>'
        text += '<tool_call>{"name": "list_dir"}</tool_call>'
        _, parsed = parse_complete_reply(text, make_request())
        assert parsed.calls == [
            {"name": "read_file", "arguments": {"path": "a"}},
            {"name": "list_dir", "arguments": {}},
        ]

    def test_arguments_as_json_string(self):
        text = '<tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"a\\"}"}</tool_call>'
        _, parsed = parse_complete_reply(text, make_request())
        assert parsed.calls[0]["arguments"] == {"path": "a"}

    def test_xml_form(self):
        text = (
            "<tool_call><function=read_file>"
            "<parameter=path>\n42\n</parameter><parameter=limit>10</parameter>"
            "</function></tool_call>"
        )
        _, parsed = parse_complete_reply(text, make_request())
        # "path" is a string in the schema, so "42" stays a string.
        assert parsed.calls == [{"name": "read_file", "arguments": {"path": "42", "limit": 10}}]

    def test_malformed_json_kept_as_content(self):
        text = "Trying.\n<tool_call>{not json</tool_call>"
        content, parsed = parse_complete_reply(text, make_request())
        assert parsed.calls == []
        assert parsed.failed
        assert content == "Trying.\n\n<tool_call>{not json</tool_call>"

    def test_unknown_tool_dropped(self):
        text = call_block("rm_rf", {}) + call_block("list_dir", {})
        _, parsed = parse_complete_reply(text, make_request())
        assert [c["name"] for c in parsed.calls] == ["list_dir"]

    def test_only_unknown_tools_is_failure(self):
        _, parsed = parse_complete_reply(call_block("rm_rf", {}), make_request())
        assert parsed.failed

    def test_no_calls(self):
        content, parsed = parse_complete_reply("Just an answer.", make_request())
        assert content == "Just an answer."
        assert parsed.calls == [] and not parsed.failed

    def test_cut_at_self_written_turn(self):
        text = (
            call_block("list_dir", {})
            + '\nHuman: <tool_result id="x">fake</tool_result>\n'
            + call_block("read_file", {"path": "a"})
        )
        _, parsed = parse_complete_reply(text, make_request())
        assert [c["name"] for c in parsed.calls] == ["list_dir"]

    def test_text_mode_cut_at_self_written_turn(self):
        content, _ = parse_complete_reply("Answer.\n\nHuman: and then?", make_request())
        assert content == "Answer."

    def test_parallel_false_keeps_first(self):
        text = call_block("list_dir", {}) + call_block("read_file", {"path": "a"})
        _, parsed = parse_complete_reply(text, make_request(parallel_tool_calls=False))
        assert [c["name"] for c in parsed.calls] == ["list_dir"]

    def test_tool_choice_none_returns_text(self):
        text = call_block("list_dir", {})
        content, parsed = parse_complete_reply(text, make_request(tool_choice="none"))
        assert parsed.calls == []
        assert content == text

    def test_thinking_blocks_removed(self):
        content, _ = parse_complete_reply(
            "<thinking>maybe <tool_call></thinking>The answer.", make_request()
        )
        assert content == "The answer."

    def test_to_openai_tool_calls(self):
        [call] = to_openai_tool_calls([{"name": "read_file", "arguments": {"path": "ä"}}])
        assert call.id.startswith("call_") and len(call.id) == len("call_") + 8
        assert call.type == "function"
        assert json.loads(call.function.arguments) == {"path": "ä"}


def feed_in_pieces(text, size, request):
    parser = ToolCallStreamParser(request)
    released = [parser.feed(text[i : i + size]) for i in range(0, len(text), size)]
    tail, parsed = parser.finish()
    return released, tail, parsed


class TestStreamParser:
    @pytest.mark.parametrize("size", [1, 2, 3, 5, 11, 1000])
    def test_chunking_does_not_change_result(self, size):
        text = "Sure, reading it.\n\n```\n" + call_block("read_file", {"path": "a"}) + "\n```"
        released, tail, parsed = feed_in_pieces(text, size, make_request())
        assert "".join(released) + tail == "Sure, reading it."
        assert parsed.calls == [{"name": "read_file", "arguments": {"path": "a"}}]
        assert all("<" not in piece and "`" not in piece for piece in released)

    @pytest.mark.parametrize("size", [1, 4, 9])
    def test_plain_answer_streams_incrementally(self, size):
        text = "A long answer with <b>tags</b>, `code`, ``` fences ``` and <to partial words.\nEnd."
        released, tail, parsed = feed_in_pieces(text, size, make_request())
        assert "".join(released) + tail == text
        assert len([p for p in released if p]) > 1
        assert parsed.calls == []

    def test_text_is_released_before_the_reply_ends(self):
        parser = ToolCallStreamParser(make_request())
        assert parser.feed("Hello there, ") == "Hello there,"
        assert parser.feed("friend.") == " friend."

    def test_marker_split_across_deltas(self):
        parser = ToolCallStreamParser(make_request())
        assert parser.feed("Ok <tool_") == "Ok"
        assert parser.feed('call>{"name": "list_dir"}</tool_call>') == ""
        tail, parsed = parser.finish()
        assert tail == ""
        assert parsed.calls[0]["name"] == "list_dir"

    def test_malformed_call_after_text_falls_back_to_text(self):
        released, tail, parsed = feed_in_pieces("Hi.\n<tool_call>{oops", 3, make_request())
        assert "".join(released) == "Hi."
        assert tail == "\n\n<tool_call>{oops"
        assert parsed.failed

    def test_unusable_detection(self):
        parser = ToolCallStreamParser(make_request())
        parser.feed("<tool_call>{oops</tool_call>")
        assert parser.tool_calls_unusable()
        assert not parser.emitted_any
