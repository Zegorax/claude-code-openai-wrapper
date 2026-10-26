"""Register the wrapper as QwenPaw's model provider (run inside the QwenPaw image).

Used by the qwenpaw-setup service in docker-compose.ghcr.yml. Safe to run on
every start: it creates the provider and models only when they are missing,
keeps the base URL and API key in sync, and sets the active model only when
none is set yet, so a model chosen later in the Console is left alone.

Environment:
    WRAPPER_BASE_URL  OpenAI base URL of the wrapper (default http://claude-wrapper:8000/v1)
    WRAPPER_API_KEY   the wrapper's API_KEY, if protection is on (any value otherwise)
    CLAUDE_MODELS     comma-separated model ids; the first becomes the active model.
                      Empty: use the wrapper's /v1/models (the models the
                      account can use, recommended model first).
"""

import asyncio
import json
import os
import urllib.request

from qwenpaw.providers.provider import ModelInfo, ProviderInfo
from qwenpaw.providers.provider_manager import ProviderManager

PROVIDER_ID = "claude-wrapper"
FALLBACK_MODEL = "claude-sonnet-5-5"


def wrapper_models(base_url: str, api_key: str) -> list:
    """Model ids from the wrapper's /v1/models, in its order."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response).get("data") or []
    except Exception as exc:  # noqa: BLE001 - fall back to one known model
        print(f"Could not read {base_url}/models ({exc}); using {FALLBACK_MODEL}")
        return [FALLBACK_MODEL]
    return [m["id"] for m in data if isinstance(m, dict) and m.get("id")] or [FALLBACK_MODEL]


async def main() -> None:
    base_url = os.environ.get("WRAPPER_BASE_URL", "http://claude-wrapper:8000/v1")
    # QwenPaw requires a key for custom providers; the wrapper ignores it
    # unless its own API_KEY is set.
    api_key = os.environ.get("WRAPPER_API_KEY") or "not-used"
    models = [m.strip() for m in os.environ.get("CLAUDE_MODELS", "").split(",") if m.strip()]
    if not models:
        models = wrapper_models(base_url, api_key)
    print(f"Models: {', '.join(models)}")

    manager = ProviderManager.get_instance()
    if manager.get_provider(PROVIDER_ID) is None:
        await manager.add_custom_provider(
            ProviderInfo(
                id=PROVIDER_ID,
                name="Claude Code (wrapper)",
                base_url=base_url,
                is_custom=True,
                chat_model="OpenAIChatModel",
            )
        )
        print(f"Created provider {PROVIDER_ID}")
    manager.update_provider(PROVIDER_ID, {"base_url": base_url, "api_key": api_key})

    provider = manager.get_provider(PROVIDER_ID)
    known = {m.id for m in [*provider.models, *provider.extra_models]}
    for model_id in models:
        if model_id not in known:
            await manager.add_model_to_provider(
                PROVIDER_ID, ModelInfo(id=model_id, name=f"{model_id} (Claude Code)")
            )
            print(f"Added model {model_id}")

    if manager.get_active_model() is None and models:
        await manager.activate_model(PROVIDER_ID, models[0])
        print(f"Active model: {PROVIDER_ID}/{models[0]}")
    else:
        print(f"Active model unchanged: {manager.get_active_model()}")


asyncio.run(main())
