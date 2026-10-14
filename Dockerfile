FROM python:3.12-slim

# Poetry version that generated poetry.lock
ENV POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Note: Claude Code CLI is bundled with claude-agent-sdk >= 0.1.8
# No separate Node.js/npm installation required

WORKDIR /app

# Install dependencies first so this layer is cached across code changes
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --no-root --only main --no-interaction \
    && rm -rf /root/.cache/pypoetry

# Copy the app code
COPY . /app

# Expose the port (default 8000)
EXPOSE 8000

# Run the app with Uvicorn. For development with hot reload, override the
# command and add --reload (see README).
CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
