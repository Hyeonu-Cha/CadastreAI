# syntax=docker/dockerfile:1.6
# Multi-stage build for the CadastreAI Streamlit + agent runtime.
#
# Stage 1 (builder) compiles wheels for the heavy deps (sentence-transformers,
# qdrant-client, anthropic, langgraph) into an isolated venv, so we don't ship
# build toolchains in the final image. Stage 2 copies the venv into a slim
# runtime image and runs Streamlit as a non-root user.
#
# Optional `parse` extras (docling, pymupdf4llm) are deliberately skipped —
# they pull torch+vision-model weights (~1-2 GB) and are only needed for the
# offline ingestion pipeline, not the agent runtime.
#
# Build:
#     docker build -t cadastreai:latest .
# Run:
#     docker run --rm -p 8501:8501 \
#         -e ANTHROPIC_API_KEY=sk-... \
#         -e QDRANT_URL=http://host.docker.internal:6333 \
#         cadastreai:latest

# ---- Stage 1: builder ------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build toolchain only — discarded in stage 2.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the dependency manifest first so Docker's layer cache reuses
# the heavy `pip install` step when source changes but deps don't.
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install project + runtime extras (no `parse`, no `dev`).
RUN pip install --upgrade pip \
    && pip install ".[agent,index,embed,tools,app,chunk]"


# ---- Stage 2: runtime ------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp1 is required by sentence-transformers / numpy at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — Streamlit doesn't need privileges, and running as root
# in a container is an avoidable security hazard.
RUN groupadd --gid 1000 cadastre \
    && useradd --uid 1000 --gid cadastre --shell /bin/bash --create-home cadastre

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=cadastre:cadastre src ./src
COPY --chown=cadastre:cadastre pyproject.toml README.md ./

# Cache dirs the agent writes to at runtime (embeddings, tool results).
RUN mkdir -p /app/data/cache/embeddings \
    && chown -R cadastre:cadastre /app/data

USER cadastre

EXPOSE 8501

# Streamlit's built-in health endpoint — returns 200 when the app is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app/streamlit_app.py"]
