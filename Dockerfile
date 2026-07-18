FROM ghcr.io/astral-sh/uv:0.11.29-python3.12-trixie-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE .python-version ./
COPY src ./src
COPY examples ./examples

RUN uv sync --frozen --no-editable

ENTRYPOINT ["uv", "run", "--no-sync", "automl"]
CMD ["--help"]
