# --- Stage 1: Build Stage ---
FROM python:3.11-slim AS builder

# Efficiently install uv using the official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment (.venv)
# --frozen ensures the lockfile is not updated during build
RUN uv sync --frozen --no-dev --no-install-project

# --- Stage 2: Runtime Stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
# This excludes uv, build tools, and package caches to keep the image slim
COPY --from=builder /app/.venv /app/.venv

# Add the virtual environment to the system PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source code
COPY . .

EXPOSE 8000

# Execute uvicorn directly for a lightweight runtime (avoids 'uv run' overhead)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]