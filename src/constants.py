"""
Constants and configuration for Claude Code OpenAI Wrapper.

Single source of truth for tool names, models, and other configuration values.

Usage Examples:
    # Get default allowed tools
    from src.constants import DEFAULT_ALLOWED_TOOLS
    options = {"allowed_tools": DEFAULT_ALLOWED_TOOLS}

    # Use rate limits in FastAPI
    from src.constants import RATE_LIMIT_CHAT
    @limiter.limit(f"{RATE_LIMIT_CHAT}/minute")
    async def chat_endpoint(): ...

Note:
    - Tool configurations are managed by ToolManager (see tool_manager.py)
    - Models are discovered at runtime, never listed here
    - Rate limits can be overridden via environment variables
"""

import os
from typing import Optional

# Claude Agent SDK Tool Names
# These are the built-in tools available in the Claude Agent SDK
# See: https://docs.anthropic.com/en/docs/claude-code/sdk
CLAUDE_TOOLS = [
    "Task",  # Launch agents for complex tasks
    "Bash",  # Execute bash commands
    "Glob",  # File pattern matching
    "Grep",  # Search file contents
    "Read",  # Read files
    "Edit",  # Edit files
    "Write",  # Write files
    "NotebookEdit",  # Edit Jupyter notebooks
    "WebFetch",  # Fetch web content
    "TodoWrite",  # Manage todo lists
    "WebSearch",  # Search the web
    "BashOutput",  # Get bash output
    "KillShell",  # Kill bash shells
    "Skill",  # Execute skills
    "SlashCommand",  # Execute slash commands
]

# Default tools to allow when tools are enabled
# Subset of CLAUDE_TOOLS that are safe and commonly used
DEFAULT_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Bash",
    "Write",
    "Edit",
]

# Tools to disallow by default (potentially dangerous or slow)
DEFAULT_DISALLOWED_TOOLS = [
    "Task",  # Can spawn sub-agents
    "WebFetch",  # External network access
    "WebSearch",  # External network access
]

# Claude Models
# Models are never hardcoded: /v1/models and the default model come from
# Anthropic's Models API (when ANTHROPIC_API_KEY is set) or from the Claude
# Code CLI, which reports the models the logged-in account can use.
#
# DEFAULT_MODEL_ENV is an optional operator choice for requests that name no
# model. When unset, the wrapper resolves a default from those sources and
# stores it in RESOLVED_DEFAULT_MODEL.
DEFAULT_MODEL_ENV: Optional[str] = os.getenv("DEFAULT_MODEL")
RESOLVED_DEFAULT_MODEL: Optional[str] = None
# The Claude Code CLI's own alias for its recommended model; used when no
# default could be resolved, so the CLI picks.
CLI_DEFAULT_MODEL_ALIAS = "default"

# Anthropic Models API configuration for dynamically refreshing /v1/models
ANTHROPIC_MODELS_URL = os.getenv("ANTHROPIC_MODELS_URL", "https://api.anthropic.com/v1/models")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
MODEL_LIST_CACHE_TTL_SECONDS = int(os.getenv("MODEL_LIST_CACHE_TTL_SECONDS", "3600"))
# Shorter TTL applied when the live fetch fails so a transient blip doesn't
# suppress live discovery for a full hour.
MODEL_LIST_ERROR_TTL_SECONDS = int(os.getenv("MODEL_LIST_ERROR_TTL_SECONDS", "60"))
MODEL_LIST_REQUEST_TIMEOUT_SECONDS = float(os.getenv("MODEL_LIST_REQUEST_TIMEOUT_SECONDS", "5"))

# System Prompt Types
SYSTEM_PROMPT_TYPE_TEXT = "text"
SYSTEM_PROMPT_TYPE_PRESET = "preset"

# System Prompt Presets
SYSTEM_PROMPT_PRESET_CLAUDE_CODE = "claude_code"

# API Configuration
DEFAULT_MAX_TURNS = 10
DEFAULT_TIMEOUT_MS = 600000  # 10 minutes
DEFAULT_PORT = 8000

# Session Management
SESSION_CLEANUP_INTERVAL_MINUTES = 5
SESSION_MAX_AGE_MINUTES = 60

# Rate Limiting (requests per minute)
RATE_LIMIT_DEFAULT = 60
RATE_LIMIT_CHAT = 30
RATE_LIMIT_MODELS = 100
RATE_LIMIT_HEALTH = 200
