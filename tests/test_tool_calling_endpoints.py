"""Endpoint tests for OpenAI function calling, with the Claude Agent SDK mocked out."""

import json
from unittest.mock import patch

import pytest
from claude_agent_sdk import TextBlock
from fastapi.testclient import TestClient

import src.main as main
from src.constants import CLAUDE_TOOLS

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}
USAGE = {
    "input_tokens": 10,
    "cache_creation_input_tokens": 5,
    "cache_read_input_tokens": 100,
    "output_tokens": 7,
}


def sdk_chunks(text, pieces=4, partial=True):
    """Chunks as ClaudeCodeCLI.run_completion yields them for one reply."""
    chunks = []
    if partial:
        size = max(1, len(text) // pieces)
        for i in range(0, len(text), size):
            chunks.append(
                {
                    "event": {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text[i : i + size]},
                    },
                    "uuid": "u",
                    "session_id": "s",
                    "parent_tool_use_id": None,
                }
            )
    chunks.append({"content": [TextBlock(text=text)], "model": "claude"})
    chunks.append(
        {
            "subtype": "success",
            "result": text,
            "is_error": False,
            "usage": USAGE,
            "duration_ms": 1200,
            "duration_api_ms": 900,
        }
    )
    return chunks


class FakeSDK:
    """Replaces claude_cli.run_completion; replies are consumed in order."""

    def __init__(self, *replies, partial=True):
        self.replies = list(replies)
        self.partial = partial
        self.calls = []
        self.chunks_consumed = 0

    async def run_completion(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        partial = self.partial and kwargs.get("include_partial_messages", False)
        for chunk in sdk_chunks(reply, pieces=20, partial=partial):
            self.chunks_consumed += 1
            yield chunk


@pytest.fixture
def fake_sdk():
    holder = {}

    def install(*replies, partial=True):
        sdk = FakeSDK(*replies, partial=partial)
        holder["patch"] = patch.object(main.claude_cli, "run_completion", sdk.run_completion)
        holder["patch"].start()
        return sdk

    with patch.object(main, "validate_claude_code_auth", return_value=(True, {})):
        yield install
    if "patch" in holder:
        holder["patch"].stop()


@pytest.fixture
def client():
    # Each test starts with a fresh per-minute chat rate limit.
    if main.limiter:
        main.limiter.reset()
    return TestClient(main.app)


def sse_events(response):
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def tool_request(**extra):
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Read a.txt"}],
        "tools": [READ_FILE],
    }
    body.update(extra)
    return body


CALL_REPLY = (
    'Reading it.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.txt"}}</tool_call>'
)


class TestNonStreaming:
    def test_tool_call_response(self, client, fake_sdk):
        sdk = fake_sdk(CALL_REPLY)
        response = client.post("/v1/chat/completions", json=tool_request())
        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] == "Reading it."
        [call] = choice["message"]["tool_calls"]
        assert call["type"] == "function"
        assert call["function"]["name"] == "read_file"
        assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}

        kwargs = sdk.calls[0]
        assert kwargs["disallowed_tools"] == CLAUDE_TOOLS
        assert kwargs["max_turns"] == 1
        # No built-in Claude Code tools are offered at all.
        assert kwargs["tools"] == []
        assert "<tools>" in kwargs["system_prompt"]

    def test_real_usage_reported(self, client, fake_sdk):
        fake_sdk("Done.")
        usage = client.post("/v1/chat/completions", json=tool_request()).json()["usage"]
        assert usage == {
            "prompt_tokens": 115,
            "completion_tokens": 7,
            "total_tokens": 122,
            "prompt_tokens_details": {"cached_tokens": 100},
        }

    def test_plain_answer(self, client, fake_sdk):
        fake_sdk("The file says hello.")
        choice = client.post("/v1/chat/completions", json=tool_request()).json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == "The file says hello."
        assert "tool_calls" not in choice["message"]

    def test_tool_history_accepted(self, client, fake_sdk):
        sdk = fake_sdk("It says hello.")
        body = tool_request(
            messages=[
                {"role": "user", "content": "Read a.txt"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "hello <bash>x</bash>"},
            ]
        )
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        prompt = sdk.calls[0]["prompt"]
        # Tool output reaches Claude unfiltered.
        assert '<tool_result id="call_1" name="read_file">\nhello <bash>x</bash>' in prompt

    def test_retry_after_unusable_call(self, client, fake_sdk):
        sdk = fake_sdk("<tool_call>{broken</tool_call>", CALL_REPLY)
        choice = client.post("/v1/chat/completions", json=tool_request()).json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert len(sdk.calls) == 2
        assert "could not be used" in sdk.calls[1]["prompt"]

    def test_no_retry_when_disabled(self, client, fake_sdk):
        sdk = fake_sdk("<tool_call>{broken</tool_call>")
        with patch("src.tool_calling.TOOL_CALL_RETRY", False):
            choice = client.post("/v1/chat/completions", json=tool_request()).json()["choices"][0]
        assert len(sdk.calls) == 1
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == "<tool_call>{broken</tool_call>"

    def test_generation_stops_at_self_written_turn(self, client, fake_sdk):
        invented = CALL_REPLY + '\n<tool_result id="x">invented output</tool_result>\n' + "x" * 400
        sdk = fake_sdk(invented)
        choice = client.post("/v1/chat/completions", json=tool_request()).json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert len(choice["message"]["tool_calls"]) == 1
        assert "invented" not in (choice["message"]["content"] or "")
        total = len(sdk_chunks(invented, pieces=20))
        assert sdk.chunks_consumed < total - 5

    def test_private_cwd_removed_from_arguments(self, client, fake_sdk):
        cwd = str(main.claude_cli.cwd)
        fake_sdk(
            '<tool_call>{"name": "read_file", "arguments": {"path": "'
            + cwd
            + '/a.txt"}}</tool_call>'
        )
        with patch.object(main.claude_cli, "temp_dir", cwd):
            choice = client.post("/v1/chat/completions", json=tool_request()).json()["choices"][0]
        args = json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])
        assert args == {"path": "a.txt"}

    def test_works_without_partial_messages(self, client, fake_sdk):
        fake_sdk(CALL_REPLY, partial=False)
        choice = client.post("/v1/chat/completions", json=tool_request()).json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"


class TestDefaultModel:
    def test_request_without_model_uses_resolved_default(self, client, fake_sdk):
        from src import constants

        sdk = fake_sdk("Hi.")
        body = tool_request()
        del body["model"]
        with (
            patch.object(constants, "DEFAULT_MODEL_ENV", None),
            patch.object(constants, "RESOLVED_DEFAULT_MODEL", "claude-sonnet-5-5"),
        ):
            data = client.post("/v1/chat/completions", json=body).json()
        assert sdk.calls[0]["model"] == "claude-sonnet-5-5"
        assert data["model"] == "claude-sonnet-5-5"

    def test_request_without_model_lets_cli_choose(self, client, fake_sdk):
        from src import constants

        async def nothing_resolved():
            return None

        sdk = fake_sdk("Hi.")
        body = tool_request()
        del body["model"]
        with (
            patch.object(constants, "DEFAULT_MODEL_ENV", None),
            patch.object(constants, "RESOLVED_DEFAULT_MODEL", None),
            patch.object(main, "resolve_default_model", nothing_resolved),
        ):
            data = client.post("/v1/chat/completions", json=body).json()
        assert sdk.calls[0]["model"] == "default"
        assert data["model"] == "default"


class TestStreaming:
    def test_chunk_sequence(self, client, fake_sdk):
        fake_sdk(CALL_REPLY)
        response = client.post(
            "/v1/chat/completions",
            json=tool_request(stream=True, stream_options={"include_usage": True}),
        )
        events = sse_events(response)
        assert events[-1] == "[DONE]"
        chunks = events[:-1]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}

        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert content == "Reading it."
        assert "<tool_call" not in content

        tool_chunks = [c for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
        [tool_chunk] = tool_chunks
        [delta_call] = tool_chunk["choices"][0]["delta"]["tool_calls"]
        assert delta_call["index"] == 0
        assert delta_call["id"].startswith("call_")
        assert delta_call["function"]["name"] == "read_file"
        assert json.loads(delta_call["function"]["arguments"]) == {"path": "a.txt"}

        final = chunks[-1]
        assert final["choices"][0]["finish_reason"] == "tool_calls"
        assert final["usage"]["total_tokens"] == 122

    def test_long_answer_streams_in_pieces(self, client, fake_sdk):
        answer = "This is a long answer. " * 20
        fake_sdk(answer)
        events = sse_events(client.post("/v1/chat/completions", json=tool_request(stream=True)))
        content_chunks = [
            e["choices"][0]["delta"]["content"]
            for e in events[:-1]
            if e["choices"][0]["delta"].get("content")
        ]
        assert len(content_chunks) > 1
        assert "".join(content_chunks) == answer.strip()
        assert events[-2]["choices"][0]["finish_reason"] == "stop"
        assert "usage" not in events[-2] or events[-2]["usage"] is None

    def test_sdk_error_becomes_error_event(self, client, fake_sdk):
        async def failing(**kwargs):
            yield {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "error_message": "boom",
            }

        with patch.object(main, "validate_claude_code_auth", return_value=(True, {})):
            with patch.object(main.claude_cli, "run_completion", failing):
                events = sse_events(
                    client.post("/v1/chat/completions", json=tool_request(stream=True))
                )
        assert events[-1]["error"]["message"] == "boom"


class TestOpenAIClientContract:
    """A full tool loop through the openai SDK, as QwenPaw would drive it."""

    @pytest.fixture
    def openai_client(self, client):
        openai = pytest.importorskip("openai")
        return openai.OpenAI(
            api_key="unused", base_url="http://testserver/v1", http_client=client, max_retries=0
        )

    def _loop(self, openai_client, stream):
        messages = [{"role": "user", "content": "What is in a.txt?"}]

        def call(**kwargs):
            if not stream:
                return openai_client.chat.completions.create(**kwargs).choices[0].message
            content, calls, finish = "", {}, None
            for chunk in openai_client.chat.completions.create(stream=True, **kwargs):
                delta = chunk.choices[0].delta
                content += delta.content or ""
                for tc in delta.tool_calls or []:
                    entry = calls.setdefault(tc.index, {"id": tc.id, "name": "", "args": ""})
                    entry["name"] += tc.function.name or ""
                    entry["args"] += tc.function.arguments or ""
                finish = chunk.choices[0].finish_reason or finish
            return content, calls, finish

        first = call(model="claude-sonnet-4-6", messages=messages, tools=[READ_FILE])
        if stream:
            content, calls, finish = first
            assert finish == "tool_calls"
            call_id, name, args = calls[0]["id"], calls[0]["name"], calls[0]["args"]
        else:
            [tc] = first.tool_calls
            call_id, name, args = tc.id, tc.function.name, tc.function.arguments
        assert name == "read_file"
        assert json.loads(args) == {"path": "a.txt"}

        messages += [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "hello"},
        ]
        second = call(model="claude-sonnet-4-6", messages=messages, tools=[READ_FILE])
        if stream:
            content, calls, finish = second
            assert finish == "stop" and calls == {}
        else:
            content = second.content
            assert second.tool_calls is None
        assert content == "a.txt says hello."

    @pytest.mark.parametrize("stream", [False, True])
    def test_tool_loop(self, openai_client, fake_sdk, stream):
        sdk = fake_sdk(CALL_REPLY, "a.txt says hello.")
        self._loop(openai_client, stream)
        assert '<tool_result id="call_' in sdk.calls[1]["prompt"]


class TestNoToolsRegression:
    """Requests without tools keep the original code path and response shape."""

    def test_non_streaming_shape(self, client, fake_sdk):
        sdk = fake_sdk("Hello!", partial=False)
        body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}]}
        data = client.post("/v1/chat/completions", json=body).json()
        assert data["choices"][0] == {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!", "name": None},
            "finish_reason": "stop",
        }
        assert "include_partial_messages" not in sdk.calls[0]

    def test_streaming_shape(self, client, fake_sdk):
        sdk = fake_sdk("Hello!", partial=False)
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        events = sse_events(client.post("/v1/chat/completions", json=body))
        deltas = [e["choices"][0]["delta"] for e in events[:-1]]
        assert deltas == [{"role": "assistant", "content": ""}, {"content": "Hello!"}, {}]
        assert events[-1] == "[DONE]"
        assert "include_partial_messages" not in sdk.calls[0]
        assert sdk.calls[0]["tools"] == []

    def test_stream_partial_flag(self, client, fake_sdk):
        sdk = fake_sdk("Hello there, this is streamed.")
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        with patch.object(main, "STREAM_PARTIAL", True):
            events = sse_events(client.post("/v1/chat/completions", json=body))
        content = [e["choices"][0]["delta"].get("content") for e in events[1:-2]]
        assert len(content) > 1
        assert "".join(content) == "Hello there, this is streamed."
        assert sdk.calls[0]["include_partial_messages"] is True
