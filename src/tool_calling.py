"""OpenAI-style function calling on top of a plain-text Claude Code turn.

Claude Code runs with its own tools disabled, so the client's tools are
described in the system prompt and Claude answers with text tags:

    <tool_call>{"name": "...", "arguments": {...}}</tool_call>

This module renders OpenAI messages into that format and parses the tags
back into native OpenAI ``tool_calls``, both for complete replies and for
replies that arrive as a stream of text deltas.
"""

import html
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

from src.models import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    Message,
    StreamChoice,
    ToolCall,
    ToolDefinition,
    Usage,
)

logger = logging.getLogger(__name__)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESULT_OPEN = "<tool_result"
# Claude sometimes continues the transcript itself; these mark where its own
# turn has ended.
HUMAN_TURN = "\n\nHuman:"
# After a tool call any "Human:" line is a self-written turn.
HUMAN_LINE = "\nHuman:"
THINKING_OPEN = "<thinking>"
THINKING_CLOSE = "</thinking>"

RETRY_NOTE = (
    "Your previous reply contained a <tool_call> block that could not be used. "
    "Reply again. Each call must be exactly "
    '<tool_call>{"name": "<function-name>", "arguments": {...}}</tool_call> '
    "with valid JSON, and must use one of the listed function names."
)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _sorted_json(value: Any) -> Any:
    """Return value with dict keys sorted, so schemas render identically every call."""
    if isinstance(value, dict):
        return {k: _sorted_json(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json(v) for v in value]
    return value


def _tool_schema_line(tool: ToolDefinition) -> str:
    fn = tool.function
    entry: Dict[str, Any] = {"name": fn.name}
    if fn.description:
        entry["description"] = fn.description
    entry["parameters"] = _sorted_json(fn.parameters or {"type": "object", "properties": {}})
    return json.dumps(entry, ensure_ascii=False)


def _forced_function_name(tool_choice: Any) -> Optional[str]:
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
    return None


def build_tools_section(request: ChatCompletionRequest) -> str:
    """System prompt section describing the request's tools and the call format."""
    lines = [
        "# Tools",
        "You can call these functions. Each tool's JSON Schema is listed below.",
        "<tools>",
        *(_tool_schema_line(t) for t in request.tools or []),
        "</tools>",
        "",
        "To call tools, reply with one or more blocks, each on its own line, exactly:",
        '<tool_call>{"name": "<function-name>", "arguments": {<args as JSON object>}}</tool_call>',
        "You may write a short sentence before the blocks. Write nothing after them.",
        "Then stop; results come back in the next message as <tool_result> blocks.",
        "If no tool is needed, answer normally without any <tool_call> block.",
    ]
    forced = _forced_function_name(request.tool_choice)
    if forced:
        lines.append(f"You must call `{forced}`.")
    elif request.tool_choice == "required":
        lines.append("You must call at least one tool.")
    if request.parallel_tool_calls is False:
        lines.append("Call at most one tool per reply.")
    return "\n".join(lines)


def _parse_arguments(raw: str) -> Any:
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return raw


def _render_tool_call(call: ToolCall) -> str:
    payload = {"name": call.function.name, "arguments": _parse_arguments(call.function.arguments)}
    return f"{TOOL_CALL_OPEN}{json.dumps(payload, ensure_ascii=False)}{TOOL_CALL_CLOSE}"


def _render_tool_result(message: Message, names_by_id: Dict[str, str]) -> str:
    call_id = message.tool_call_id or ""
    name = names_by_id.get(call_id) or message.name
    attrs = f' id="{html.escape(call_id, quote=True)}"'
    if name:
        attrs += f' name="{html.escape(name, quote=True)}"'
    return f"<tool_result{attrs}>\n{message.content or ''}\n</tool_result>"


def render_prompt(
    request: ChatCompletionRequest, messages: Optional[List[Message]] = None
) -> Tuple[str, Optional[str]]:
    """Render the request into (prompt, system_prompt) for the tool-calling path.

    messages defaults to request.messages (pass the session-merged list when
    session_id is used).
    """
    messages = request.messages if messages is None else messages
    system_parts: List[str] = []
    turns: List[str] = []
    names_by_id: Dict[str, str] = {}
    pending_results: List[str] = []

    def flush_results() -> None:
        if pending_results:
            turns.append("Human: " + "\n".join(pending_results))
            pending_results.clear()

    for message in messages:
        if message.role != "tool":
            flush_results()

        if message.role in ("system", "developer"):
            if message.content:
                system_parts.append(message.content)  # type: ignore[arg-type]
        elif message.role == "user":
            turns.append(f"Human: {message.content or ''}")
        elif message.role == "assistant":
            parts = [message.content] if message.content else []
            for call in message.tool_calls or []:
                names_by_id[call.id] = call.function.name
                parts.append(_render_tool_call(call))
            if parts:
                turns.append("Assistant: " + "\n".join(parts))  # type: ignore[arg-type]
        elif message.role == "tool":
            pending_results.append(_render_tool_result(message, names_by_id))
    flush_results()

    prompt = "\n\n".join(turns)
    if messages and messages[-1].role == "assistant":
        prompt += "\n\nHuman: Please continue."

    if request.tool_calling_enabled():
        system_parts.append(build_tools_section(request))
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return prompt, system_prompt


def retry_prompt(prompt: str, failed_reply: str) -> str:
    """Prompt for one more attempt after a reply whose tool calls could not be used."""
    return f"{prompt}\n\nAssistant: {failed_reply}\n\nHuman: {RETRY_NOTE}"


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)
_XML_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|$)", re.DOTALL)
_XML_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
# An unfinished code fence opener ("```" or "```json") with nothing after it yet.
_TRAILING_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*\s*$")
# The same, or the first backticks of one, while a reply is still streaming.
_PARTIAL_FENCE_RE = re.compile(r"(?:```[A-Za-z0-9_-]*|``?)\s*$")


def _strip_fence(body: str) -> str:
    match = _FENCE_RE.match(body.strip())
    return match.group(1).strip() if match else body.strip()


def _parse_xml_call(body: str, schemas: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    match = _XML_FUNCTION_RE.search(body)
    if not match:
        return None
    name = match.group(1).strip()
    properties = (schemas.get(name) or {}).get("properties") or {}
    args: Dict[str, Any] = {}
    for key, raw in _XML_PARAMETER_RE.findall(match.group(2)):
        value = raw.strip("\n")
        prop_type = (properties.get(key) or {}).get("type")
        if prop_type != "string":
            try:
                args[key] = json.loads(value)
                continue
            except ValueError:
                pass
        args[key] = value
    return {"name": name, "arguments": args}


def _parse_json_call(body: str) -> Optional[Dict[str, Any]]:
    obj: Any = None
    try:
        obj = json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(body[start : end + 1])
            except ValueError:
                return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return {"name": obj["name"], "arguments": args}


def parse_call_body(body: str, schemas: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Parse the inside of one <tool_call> block, or return None if it is unusable."""
    body = _strip_fence(body)
    if body.startswith("<function="):
        return _parse_xml_call(body, schemas)
    return _parse_json_call(body)


@dataclass
class ParsedToolCalls:
    calls: List[Dict[str, Any]] = field(default_factory=list)
    # True when the text contained tool-call markup but no usable call.
    failed: bool = False


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def parse_tool_section(text: str, request: ChatCompletionRequest) -> ParsedToolCalls:
    """Parse the part of a reply that starts at its first <tool_call> tag.

    Stops at the first sign that Claude went on to write the next turn itself
    (a "Human:" turn or a <tool_result> block). Calls to tools the request
    did not offer are dropped.
    """
    schemas = {t.function.name: t.function.parameters or {} for t in request.tools or []}
    result = ParsedToolCalls()
    pos = 0
    while True:
        start = text.find(TOOL_CALL_OPEN, pos)
        leak = min(
            (i for i in (text.find(HUMAN_LINE, pos), text.find(TOOL_RESULT_OPEN, pos)) if i != -1),
            default=-1,
        )
        if start == -1 or (leak != -1 and leak < start):
            break
        body_start = start + len(TOOL_CALL_OPEN)
        end = text.find(TOOL_CALL_CLOSE, body_start)
        if end == -1:
            # Unterminated block: use what is there, up to any leaked turn.
            body = text[body_start:] if leak == -1 else text[body_start:leak]
            next_pos = len(text)
        else:
            body = text[body_start:end]
            next_pos = end + len(TOOL_CALL_CLOSE)

        call = parse_call_body(body, schemas)
        if call is None:
            logger.warning(f"Ignoring malformed tool call: {body[:200]!r}")
        elif call["name"] not in schemas:
            logger.warning(f"Ignoring call to unknown tool {call['name']!r}")
        else:
            result.calls.append(call)
        pos = next_pos

    if request.parallel_tool_calls is False and len(result.calls) > 1:
        logger.warning(f"Keeping 1 of {len(result.calls)} tool calls (parallel_tool_calls=false)")
        result.calls = result.calls[:1]
    result.failed = not result.calls
    return result


def to_openai_tool_calls(calls: List[Dict[str, Any]]) -> List[ToolCall]:
    return [
        ToolCall.model_validate(
            {
                "id": _new_call_id(),
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                },
            }
        )
        for call in calls
    ]


# ---------------------------------------------------------------------------
# Incremental parsing of a streamed reply
# ---------------------------------------------------------------------------


def _held_suffix_length(text: str, markers: List[str]) -> int:
    """Length of the longest suffix of text that could be the start of a marker."""
    longest = 0
    for marker in markers:
        for size in range(min(len(marker) - 1, len(text)), longest, -1):
            if marker.startswith(text[-size:]):
                longest = size
                break
    return longest


class ToolCallStreamParser:
    """Splits a reply, delivered as text deltas, into safe text and tool calls.

    Text is released as soon as it cannot be the start of a <tool_call> tag
    (or of a leaked "Human:" turn, a <tool_result> block or a <thinking>
    block). From the first <tool_call> on, everything is buffered and parsed
    by finish(). The same parser handles complete replies: feed() the whole
    text once, then call finish().
    """

    def __init__(self, request: ChatCompletionRequest):
        self.request = request
        self.parse_calls = request.tool_calling_enabled()
        self._markers = [HUMAN_TURN, TOOL_RESULT_OPEN, THINKING_OPEN]
        if self.parse_calls:
            self._markers.insert(0, TOOL_CALL_OPEN)
        self._pending = ""
        self._tool_text = ""
        self._mode = "text"  # "text" | "tools" | "stopped"
        self._emitted = ""  # text released so far
        self._parsed: Optional[ParsedToolCalls] = None
        self.raw = ""  # everything fed, for retries and session storage

    @property
    def in_tool_section(self) -> bool:
        return self._mode == "tools"

    @property
    def emitted_any(self) -> bool:
        return bool(self._emitted)

    @property
    def content(self) -> str:
        """All text released so far (the message content once finished)."""
        return self._emitted

    def _parsed_tool_section(self) -> ParsedToolCalls:
        if self._parsed is None:
            self._parsed = parse_tool_section(self._tool_text, self.request)
        return self._parsed

    def tool_calls_unusable(self) -> bool:
        """True when the reply has tool-call markup but no usable call."""
        return self.in_tool_section and self._parsed_tool_section().failed

    def feed(self, delta: str) -> str:
        """Add a text delta; return the text that is now safe to send as content."""
        self.raw += delta
        if self._mode == "tools":
            self._tool_text += delta
            return ""
        if self._mode == "stopped":
            return ""

        self._pending += delta
        out: List[str] = []
        while True:
            hits = [(self._pending.find(m), m) for m in self._markers]
            hits = [(i, m) for i, m in hits if i != -1]
            if not hits:
                break
            index, marker = min(hits)
            out.append(self._pending[:index])
            rest = self._pending[index:]
            if marker == THINKING_OPEN:
                close = rest.find(THINKING_CLOSE)
                if close == -1:
                    # Wait for the end of the thinking block.
                    self._pending = rest
                    return self._release("".join(out))
                self._pending = rest[close + len(THINKING_CLOSE) :]
                continue
            self._pending = ""
            if marker == TOOL_CALL_OPEN:
                self._mode = "tools"
                self._tool_text = rest
            else:
                logger.info("Cutting reply at a self-written next turn")
                self._mode = "stopped"
            return self._release("".join(out), final=True)

        # Hold back a possible partial marker, the whitespace before it (so
        # content before a tool call can be trimmed) and, when tools are on,
        # a code fence opener that may be wrapping an upcoming <tool_call>.
        cut = len(self._pending) - _held_suffix_length(self._pending, self._markers)
        fence = _PARTIAL_FENCE_RE.search(self._pending[:cut])
        if fence and self.parse_calls:
            cut = fence.start()
        cut = len(self._pending[:cut].rstrip())
        out.append(self._pending[:cut])
        self._pending = self._pending[cut:]
        return self._release("".join(out))

    def _release(self, text: str, final: bool = False) -> str:
        if not self._emitted:
            text = text.lstrip()
        if final:
            text = text.rstrip()
            # Drop a code fence opened just before the tool call, but not one
            # that closes a code block.
            fence = _TRAILING_FENCE_RE.search(text)
            if fence and (self._emitted + text).count("```") % 2 == 1:
                text = text[: fence.start()].rstrip()
        self._emitted += text
        return text

    def finish(self) -> Tuple[str, ParsedToolCalls]:
        """Return (remaining text to send, parsed tool calls) once the reply is complete."""
        if self._mode == "tools":
            parsed = self._parsed_tool_section()
            if parsed.failed:
                # Keep the unusable markup as plain text rather than losing it.
                text = self._tool_text.strip()
                if self._emitted:
                    text = "\n\n" + text
                self._emitted += text
                return text, parsed
            return "", parsed
        remaining = self._pending if self._mode == "text" else ""
        self._pending = ""
        return self._release(remaining).rstrip(), ParsedToolCalls()


def parse_complete_reply(text: str, request: ChatCompletionRequest) -> Tuple[str, ParsedToolCalls]:
    """Parse a complete reply into (content, tool calls)."""
    parser = ToolCallStreamParser(request)
    head = parser.feed(text)
    tail, parsed = parser.finish()
    return head + tail, parsed


# ---------------------------------------------------------------------------
# Running a completion
# ---------------------------------------------------------------------------

TOOL_CALL_RETRY = os.getenv("TOOL_CALL_RETRY", "true").lower() in ("true", "1", "yes", "on")


def _assistant_text(chunk: Dict[str, Any]) -> Optional[str]:
    """Text of a complete assistant message chunk, if it is one."""
    content: Any = None
    if chunk.get("type") == "assistant" and isinstance(chunk.get("message"), dict):
        content = chunk["message"].get("content")
    elif isinstance(chunk.get("content"), list):
        content = chunk["content"]
    if content is None:
        return None
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts) if parts else None


def _text_delta(chunk: Dict[str, Any]) -> Optional[str]:
    """Text of a top-level text_delta stream event, if the chunk is one."""
    event = chunk.get("event")
    if not isinstance(event, dict) or chunk.get("parent_tool_use_id"):
        return None
    if event.get("type") != "content_block_delta":
        return None
    delta = event.get("delta") or {}
    if delta.get("type") == "text_delta":
        return delta.get("text", "")
    return None


@dataclass
class AttemptResult:
    parser: ToolCallStreamParser
    usage: Optional[Dict[str, Any]] = None


async def _run_attempt(
    cli: Any,
    prompt: str,
    system_prompt: Optional[str],
    options: Dict[str, Any],
    parser: ToolCallStreamParser,
    result: AttemptResult,
):
    """Run one SDK call, feeding its text to parser and yielding releasable text."""
    saw_delta = False
    fed_any = False
    async for chunk in cli.run_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        model=options.get("model"),
        max_turns=options.get("max_turns", 1),
        disallowed_tools=options.get("disallowed_tools"),
        stream=True,
        include_partial_messages=True,
    ):
        if chunk.get("is_error") or chunk.get("subtype") == "error_during_execution":
            message = chunk.get("error_message") or chunk.get("result") or "Claude Code error"
            raise RuntimeError(message)

        delta = _text_delta(chunk)
        if delta is not None:
            saw_delta = True
            fed_any = True
            text = parser.feed(delta)
            if text:
                yield text
            continue

        if "event" in chunk:
            continue

        if chunk.get("subtype") == "success" and "result" in chunk:
            result.usage = chunk.get("usage") if isinstance(chunk.get("usage"), dict) else None
            logger.info(
                "Tool-calling step: duration_ms=%s duration_api_ms=%s usage=%s",
                chunk.get("duration_ms"),
                chunk.get("duration_api_ms"),
                result.usage,
            )
            if not fed_any and chunk.get("result"):
                text = parser.feed(chunk["result"])
                if text:
                    yield text
            continue

        # Without partial messages (older CLIs) only whole messages arrive.
        full = _assistant_text(chunk)
        if full and not saw_delta:
            fed_any = True
            text = parser.feed(full)
            if text:
                yield text


def _usage(prompt: str, completion: str, sdk_usage: Optional[Dict[str, Any]]) -> Usage:
    if sdk_usage and "input_tokens" in sdk_usage:
        prompt_tokens = int(
            sdk_usage.get("input_tokens", 0)
            + (sdk_usage.get("cache_creation_input_tokens") or 0)
            + (sdk_usage.get("cache_read_input_tokens") or 0)
        )
        completion_tokens = int(sdk_usage.get("output_tokens", 0))
    else:
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _should_retry(parser: ToolCallStreamParser, attempt: int) -> bool:
    return (
        TOOL_CALL_RETRY and attempt == 0 and not parser.emitted_any and parser.tool_calls_unusable()
    )


async def run_tool_completion(
    cli: Any,
    request: ChatCompletionRequest,
    prompt: str,
    system_prompt: Optional[str],
    options: Dict[str, Any],
) -> Tuple[Message, str, Usage]:
    """Non-streaming completion: return (assistant message, finish_reason, usage)."""
    for attempt in range(2):
        parser = ToolCallStreamParser(request)
        result = AttemptResult(parser)
        head = "".join(
            [t async for t in _run_attempt(cli, prompt, system_prompt, options, parser, result)]
        )
        if _should_retry(parser, attempt):
            logger.warning("Retrying once after an unusable tool call")
            prompt = retry_prompt(prompt, parser.raw)
            continue
        tail, parsed = parser.finish()
        break

    content = head + tail
    tool_calls = to_openai_tool_calls(parsed.calls) if parsed.calls else None
    message = Message(role="assistant", content=content or None, tool_calls=tool_calls)
    finish_reason = "tool_calls" if tool_calls else "stop"
    return message, finish_reason, _usage(prompt, parser.raw, result.usage)


def _sse(request_id: str, model: str, delta: Dict[str, Any], finish_reason=None, usage=None) -> str:
    chunk = ChatCompletionStreamResponse(
        id=request_id,
        model=model,
        choices=[StreamChoice(index=0, delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )
    return f"data: {chunk.model_dump_json()}\n\n"


async def stream_tool_completion(
    cli: Any,
    request: ChatCompletionRequest,
    request_id: str,
    prompt: str,
    system_prompt: Optional[str],
    options: Dict[str, Any],
    on_complete: Optional[Callable[[Message], None]] = None,
) -> AsyncGenerator[str, None]:
    """Streaming completion as SSE lines: text as it arrives, then any tool calls."""
    model = request.model
    yield _sse(request_id, model, {"role": "assistant", "content": ""})

    for attempt in range(2):
        parser = ToolCallStreamParser(request)
        result = AttemptResult(parser)
        async for text in _run_attempt(cli, prompt, system_prompt, options, parser, result):
            yield _sse(request_id, model, {"content": text})
        if _should_retry(parser, attempt):
            logger.warning("Retrying once after an unusable tool call")
            prompt = retry_prompt(prompt, parser.raw)
            continue
        break

    tail, parsed = parser.finish()
    if tail:
        yield _sse(request_id, model, {"content": tail})

    tool_calls = to_openai_tool_calls(parsed.calls) if parsed.calls else []
    for index, call in enumerate(tool_calls):
        delta = {"tool_calls": [{"index": index, **call.model_dump()}]}
        yield _sse(request_id, model, delta)

    if on_complete:
        content = parser.content
        on_complete(
            Message(role="assistant", content=content or None, tool_calls=tool_calls or None)
        )

    usage = None
    if request.stream_options and request.stream_options.include_usage:
        usage = _usage(prompt, parser.raw, result.usage)
    finish_reason = "tool_calls" if tool_calls else "stop"
    yield _sse(request_id, model, {}, finish_reason=finish_reason, usage=usage)
    yield "data: [DONE]\n\n"
