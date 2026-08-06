FROM python:3.11-slim

# Install system dependencies
# - default-jre: Required by JPype1 and py-tetrad for Java algorithms
# - git: Required by uv to fetch git repositories
# - build-essential: Required for compiling some C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Create an empty README.md because pyproject.toml requires it,
# but we don't want changes to the real README.md to invalidate the cache.
RUN touch README.md

# Sync dependencies without installing the project itself yet
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --frozen --no-install-project

# Copy the rest of the application
COPY README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Install the project itself
RUN uv sync --frozen

# Add the virtual environment to PATH so that python commands run inside it
ENV PATH="/app/.venv/bin:$PATH"

# Use the CLI tool as the entrypoint
ENTRYPOINT ["bayesian-learn"]
# Default argument
CMD ["--help"]
