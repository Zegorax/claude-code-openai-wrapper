"""Unit tests for dynamic Anthropic model listing."""

import asyncio

import pytest

from src import constants, main


@pytest.fixture(autouse=True)
def no_cli_model_discovery(monkeypatch):
    """Keep tests from starting a real Claude Code CLI; tests opt in by patching."""

    async def no_cli_models(*args, **kwargs):
        return None

    monkeypatch.setattr(main.claude_cli, "list_models", no_cli_models)
    monkeypatch.setattr(main, "_cli_default_model", None)


@pytest.mark.asyncio
async def test_get_available_models_uses_anthropic_models_api(monkeypatch):
    main._model_list_cache = {"expires_at": 0.0, "models": None}

    async def fake_fetch():
        return [
            {
                "id": "claude-test-latest",
                "object": "model",
                "owned_by": "anthropic",
                "display_name": "Claude Test Latest",
            }
        ]

    monkeypatch.delenv("CLAUDE_MODELS_OVERRIDE", raising=False)
    monkeypatch.setattr(main, "_fetch_anthropic_models", fake_fetch)

    models = await main.get_available_models()

    assert models[0]["id"] == "claude-test-latest"
    assert models[0]["display_name"] == "Claude Test Latest"


@pytest.mark.asyncio
async def test_get_available_models_falls_back_to_constants(monkeypatch):
    main._model_list_cache = {"expires_at": 0.0, "models": None}

    async def fake_fetch():
        return None

    monkeypatch.delenv("CLAUDE_MODELS_OVERRIDE", raising=False)
    monkeypatch.setattr(main, "_fetch_anthropic_models", fake_fetch)

    models = await main.get_available_models()

    assert {model["id"] for model in models} >= {"claude-sonnet-4-6", "claude-opus-4-6"}


@pytest.mark.asyncio
async def test_model_override_skips_live_fetch(monkeypatch):
    main._model_list_cache = {"expires_at": 0.0, "models": None}

    async def fake_fetch():
        raise AssertionError("override should not call live Anthropic API")

    monkeypatch.setenv("CLAUDE_MODELS_OVERRIDE", "custom-a,custom-b")
    monkeypatch.setattr(main, "CLAUDE_MODELS", ["custom-a", "custom-b"])
    monkeypatch.setattr(main, "_fetch_anthropic_models", fake_fetch)

    models = await main.get_available_models()

    assert [model["id"] for model in models] == ["custom-a", "custom-b"]


def test_openai_model_from_anthropic_preserves_metadata():
    model = main._openai_model_from_anthropic(
        {
            "id": "claude-test",
            "type": "model",
            "display_name": "Claude Test",
            "created_at": "2026-01-01T00:00:00Z",
            "max_input_tokens": 200000,
            "max_tokens": 64000,
            "capabilities": {"batch": {"supported": True}},
        }
    )

    assert model["id"] == "claude-test"
    assert model["object"] == "model"
    assert model["owned_by"] == "anthropic"
    # `created` should be the unix timestamp of the ISO `created_at`.
    assert model["created"] == 1767225600
    assert model["capabilities"] == {"batch": {"supported": True}}


def test_fallback_objects_include_created_field():
    fallback = main._fallback_model_payload()

    assert fallback, "fallback list should not be empty"
    for entry in fallback:
        assert isinstance(entry["created"], int) and entry["created"] > 0


@pytest.mark.asyncio
async def test_concurrent_calls_only_fetch_once(monkeypatch):
    """Lock + double-check should prevent thundering-herd on cache expiry."""
    main._model_list_cache = {"expires_at": 0.0, "models": None}
    call_count = 0

    async def fake_fetch():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return [{"id": "claude-test", "object": "model", "owned_by": "anthropic"}]

    monkeypatch.delenv("CLAUDE_MODELS_OVERRIDE", raising=False)
    monkeypatch.setattr(main, "_fetch_anthropic_models", fake_fetch)

    results = await asyncio.gather(*[main.get_available_models() for _ in range(8)])

    assert call_count == 1
    for r in results:
        assert r[0]["id"] == "claude-test"


@pytest.mark.asyncio
async def test_failed_fetch_uses_short_error_ttl(monkeypatch):
    main._model_list_cache = {"expires_at": 0.0, "models": None}

    async def fake_fetch():
        return None

    monkeypatch.delenv("CLAUDE_MODELS_OVERRIDE", raising=False)
    monkeypatch.setattr(main, "_fetch_anthropic_models", fake_fetch)
    monkeypatch.setattr(main, "MODEL_LIST_CACHE_TTL_SECONDS", 3600)
    monkeypatch.setattr(main, "MODEL_LIST_ERROR_TTL_SECONDS", 60)

    await main.get_available_models()

    expires_at = main._model_list_cache["expires_at"]
    # Error TTL ~60s; success TTL ~3600s. Confirm we used the short one.
    import time as _time

    assert expires_at - _time.time() < 120


def test_pick_latest_sonnet_prefers_newest_created_at():
    models = [
        {"id": "claude-sonnet-4-5", "created_at": "2025-09-29T00:00:00Z"},
        {"id": "claude-sonnet-4-6", "created_at": "2026-04-01T00:00:00Z"},
        {"id": "claude-opus-4-6", "created_at": "2026-04-15T00:00:00Z"},
    ]

    assert main._pick_latest_sonnet(models) == "claude-sonnet-4-6"


def test_pick_latest_sonnet_returns_none_when_no_sonnet():
    models = [{"id": "claude-haiku-4-5", "created_at": "2025-10-01T00:00:00Z"}]

    assert main._pick_latest_sonnet(models) is None


@pytest.mark.asyncio
async def test_resolve_default_model_sets_constants(monkeypatch):
    main._model_list_cache = {"expires_at": 0.0, "models": None}
    constants.RESOLVED_DEFAULT_MODEL = None

    async def fake_fetch():
        return [
            {
                "id": "claude-sonnet-4-7",
                "object": "model",
                "owned_by": "anthropic",
                "created_at": "2026-06-01T00:00:00Z",
            },
            {
                "id": "claude-sonnet-4-6",
                "object": "model",
                "owned_by": "anthropic",
                "created_at": "2026-04-01T00:00:00Z",
            },
        ]

    monkeypatch.delenv("CLAUDE_MODELS_OVERRIDE", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(constants, "DEFAULT_MODEL_ENV", None)
    monkeypatch.setattr(main, "_fetch_anthropic_models", fake_fetch)

    resolved = await main.resolve_default_model()

    assert resolved == "claude-sonnet-4-7"
    assert constants.RESOLVED_DEFAULT_MODEL == "claude-sonnet-4-7"


@pytest.mark.asyncio
async def test_resolve_default_model_skips_without_api_key(monkeypatch, caplog):
    """No ANTHROPIC_API_KEY -> skip live discovery, log clearly, use fallback."""
    constants.RESOLVED_DEFAULT_MODEL = None

    async def fake_fetch():
        raise AssertionError("should not call live API without ANTHROPIC_API_KEY")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(constants, "DEFAULT_MODEL_ENV", None)
    monkeypatch.setattr(main, "_fetch_anthropic_models", fake_fetch)

    with caplog.at_level("INFO", logger="src.main"):
        resolved = await main.resolve_default_model()

    assert resolved is None
    assert constants.RESOLVED_DEFAULT_MODEL is None
    assert any("Live model discovery disabled" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_resolve_default_model_honors_env_override(monkeypatch):
    main._model_list_cache = {"expires_at": 0.0, "models": None}
    constants.RESOLVED_DEFAULT_MODEL = None

    async def fake_fetch():
        raise AssertionError("env override should short-circuit fetch")

    monkeypatch.setattr(constants, "DEFAULT_MODEL_ENV", "claude-opus-4-6")
    monkeypatch.setattr(main, "_fetch_anthropic_models", fake_fetch)

    resolved = await main.resolve_default_model()

    assert resolved == "claude-opus-4-6"
    assert constants.RESOLVED_DEFAULT_MODEL is None


def test_get_default_model_prefers_resolved_over_fallback(monkeypatch):
    from src import models as models_module

    monkeypatch.setattr(constants, "DEFAULT_MODEL_ENV", None)
    monkeypatch.setattr(constants, "RESOLVED_DEFAULT_MODEL", "claude-sonnet-future")

    assert models_module.get_default_model() == "claude-sonnet-future"


def test_get_default_model_env_override_wins(monkeypatch):
    from src import models as models_module

    monkeypatch.setattr(constants, "DEFAULT_MODEL_ENV", "claude-opus-4-6")
    monkeypatch.setattr(constants, "RESOLVED_DEFAULT_MODEL", "claude-sonnet-future")

    assert models_module.get_default_model() == "claude-opus-4-6"


CLI_MODEL_ENTRIES = [
    {"value": "default", "resolvedModel": "claude-sonnet-5-5", "displayName": "Default"},
    {"value": "opus", "resolvedModel": "claude-opus-5-5", "displayName": "Opus 5.5"},
    {"value": "sonnet", "resolvedModel": "claude-sonnet-5-5", "displayName": "Sonnet 5.5"},
    {"value": "haiku", "resolvedModel": "claude-haiku-5-5", "displayName": "Haiku 5.5"},
    {"value": "claude-opus-4-8", "resolvedModel": "claude-opus-4-8", "displayName": "Opus 4.8"},
    {"value": "claude-sonnet-4-6", "displayName": "Sonnet 4.6", "description": "Older"},
    {"value": "claude-hidden-1", "displayName": "Hidden", "disabled": True},
    {"value": "not-a-claude-id", "displayName": "Odd"},
]


def test_openai_models_from_cli():
    models, default_model = main._openai_models_from_cli(CLI_MODEL_ENTRIES)

    assert default_model == "claude-sonnet-5-5"
    # The recommended (default) model comes first.
    assert [m["id"] for m in models] == [
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-haiku-5-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
    ]
    sonnet = models[0]
    assert sonnet["display_name"] == "Sonnet 5.5"
    assert sonnet["object"] == "model" and sonnet["owned_by"] == "anthropic"
    assert isinstance(sonnet["created"], int)
    assert models[-1]["description"] == "Older"


def test_openai_models_from_cli_adds_unlisted_default():
    models, default_model = main._openai_models_from_cli(
        [{"value": "default", "resolvedModel": "claude-new-1"}, {"value": "claude-x-1"}]
    )
    assert default_model == "claude-new-1"
    assert [m["id"] for m in models] == ["claude-new-1", "claude-x-1"]


@pytest.mark.asyncio
async def test_get_available_models_uses_cli_without_api_key(monkeypatch):
    main._model_list_cache = {"expires_at": 0.0, "models": None}

    async def no_api_models():
        return None

    async def cli_entries(*args, **kwargs):
        return CLI_MODEL_ENTRIES

    monkeypatch.delenv("CLAUDE_MODELS_OVERRIDE", raising=False)
    monkeypatch.setattr(main, "_fetch_anthropic_models", no_api_models)
    monkeypatch.setattr(main.claude_cli, "list_models", cli_entries)

    models = await main.get_available_models()

    assert models[0]["id"] == "claude-sonnet-5-5"
    assert "claude-opus-4-8" in {m["id"] for m in models}
    assert main._cli_default_model == "claude-sonnet-5-5"
    # A successful listing is cached like a live one.
    assert main._model_list_cache["expires_at"] > main.time.time() + 60


@pytest.mark.asyncio
async def test_cli_discovery_can_be_disabled(monkeypatch):
    async def must_not_run(*args, **kwargs):
        raise AssertionError("CLI discovery is disabled")

    monkeypatch.setattr(main, "CLI_MODEL_DISCOVERY", False)
    monkeypatch.setattr(main.claude_cli, "list_models", must_not_run)

    assert await main._fetch_cli_models() is None


@pytest.mark.asyncio
async def test_resolve_default_model_uses_cli_default_without_api_key(monkeypatch):
    main._model_list_cache = {"expires_at": 0.0, "models": None}
    constants.RESOLVED_DEFAULT_MODEL = None

    async def cli_models():
        main._cli_default_model = "claude-sonnet-5-5"
        return [
            {"id": "claude-sonnet-5-5", "object": "model", "created": 0, "owned_by": "anthropic"}
        ]

    async def no_api_models():
        return None

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODELS_OVERRIDE", raising=False)
    monkeypatch.setattr(constants, "DEFAULT_MODEL_ENV", None)
    monkeypatch.setattr(main, "_fetch_anthropic_models", no_api_models)
    monkeypatch.setattr(main, "_fetch_cli_models", cli_models)

    try:
        resolved = await main.resolve_default_model()
        assert resolved == "claude-sonnet-5-5"
        assert constants.RESOLVED_DEFAULT_MODEL == "claude-sonnet-5-5"
    finally:
        constants.RESOLVED_DEFAULT_MODEL = None
