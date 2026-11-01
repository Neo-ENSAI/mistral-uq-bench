FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "mistral-conformal-eval"]
CMD ["--engine", "mock", "--alpha", "0.05"]
