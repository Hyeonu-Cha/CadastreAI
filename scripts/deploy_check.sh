#!/usr/bin/env bash
# scripts/deploy_check.sh — one-shot pre-launch verification.
#
# Task 4.16 wrapper. Runs after Tasks 4.14 (Qdrant Cloud upsert) and
# 4.15 (HF Spaces / Modal deploy) to verify the deployed pipeline is
# wired correctly end-to-end before announcing the URL.
#
# What it does:
#   1. Checks that all required env vars are set (fails fast with a
#      clear message if not — saves a confusing Python traceback).
#   2. Pings the Qdrant cluster and verifies the `cadastre_chunks`
#      collection exists with the expected point count (~42k).
#   3. Runs scripts/smoke_prod.py — the 5-query smoke test.
#
# Exit codes mirror smoke_prod.py:
#   0  — all checks passed
#   1  — one or more smoke queries failed
#   2  — environment misconfigured OR Qdrant unreachable / collection missing
#
# Usage:
#   QDRANT_URL=https://<cluster>.qdrant.io \
#   QDRANT_API_KEY=<key> \
#   ANTHROPIC_API_KEY=sk-... \
#   ./scripts/deploy_check.sh
#
# Or with a .env file:
#   set -a; source .env; set +a; ./scripts/deploy_check.sh

set -euo pipefail

# Colors: only emit if stdout is a terminal, so log files stay clean.
if [[ -t 1 ]]; then
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'
    BOLD=$'\033[1m'
    RESET=$'\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' RESET=''
fi

step() { printf '%s==>%s %s\n' "${BLUE}${BOLD}" "${RESET}" "$1"; }
ok()   { printf '%s ✓ %s%s\n' "${GREEN}" "$1" "${RESET}"; }
warn() { printf '%s ! %s%s\n' "${YELLOW}" "$1" "${RESET}"; }
fail() { printf '%s ✗ %s%s\n' "${RED}" "$1" "${RESET}"; }

EXPECTED_COLLECTION="${QDRANT_COLLECTION:-cadastre_chunks}"
# Loose lower bound — the corpus is 41,959 chunks. Anything well below
# that means the upsert was incomplete or hit a different collection.
MIN_POINTS=40000

# ---- Step 1: required env vars ---------------------------------------
step "Checking required environment variables"

missing=()
for var in QDRANT_URL ANTHROPIC_API_KEY; do
    if [[ -z "${!var:-}" ]]; then
        missing+=("$var")
    fi
done

# QDRANT_API_KEY isn't strictly required for an open localhost cluster
# but is mandatory for any cloud Qdrant. Warn (not fail) on the URL
# heuristic so a deliberately-open local check still passes.
if [[ -z "${QDRANT_API_KEY:-}" ]]; then
    case "${QDRANT_URL:-}" in
        http://localhost:*|http://127.0.0.1:*)
            warn "QDRANT_API_KEY unset — OK for a local open cluster, MUST be set for Qdrant Cloud"
            ;;
        *)
            missing+=("QDRANT_API_KEY")
            ;;
    esac
fi

if (( ${#missing[@]} > 0 )); then
    fail "missing required env vars: ${missing[*]}"
    exit 2
fi
ok "QDRANT_URL=${QDRANT_URL}"
ok "ANTHROPIC_API_KEY set (${#ANTHROPIC_API_KEY} chars)"
[[ -n "${QDRANT_API_KEY:-}" ]] && ok "QDRANT_API_KEY set"

# ---- Step 2: Qdrant reachability + collection sanity -----------------
step "Probing Qdrant collection ${EXPECTED_COLLECTION}"

# Inline Python — keeps the script self-contained and avoids shipping
# a second tiny helper module. The qdrant-client dep is already in
# the [index] extra so any dev environment that ran the upsert has it.
qdrant_check_output=$(python - <<'PY'
import os
import sys

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("IMPORT_ERROR")
    sys.exit(0)

url = os.environ["QDRANT_URL"]
api_key = os.environ.get("QDRANT_API_KEY") or None
collection = os.environ.get("QDRANT_COLLECTION", "cadastre_chunks")

try:
    client = QdrantClient(url=url, api_key=api_key, timeout=10)
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        print(f"MISSING\t{','.join(sorted(existing)) or '<none>'}")
        sys.exit(0)
    count = client.count(collection_name=collection, exact=True).count
    print(f"OK\t{count}")
except Exception as exc:
    print(f"ERROR\t{type(exc).__name__}: {exc}")
PY
) || { fail "qdrant probe subprocess failed"; exit 2; }

case "$qdrant_check_output" in
    IMPORT_ERROR)
        fail "qdrant-client not installed; run: pip install -e '.[index]'"
        exit 2
        ;;
    OK*)
        count=$(printf '%s' "$qdrant_check_output" | cut -f2)
        if (( count < MIN_POINTS )); then
            fail "collection has only ${count} points — expected >= ${MIN_POINTS} (re-run the upsert?)"
            exit 2
        fi
        ok "collection ${EXPECTED_COLLECTION} reachable, ${count} points"
        ;;
    MISSING*)
        existing=$(printf '%s' "$qdrant_check_output" | cut -f2)
        fail "collection ${EXPECTED_COLLECTION} not found at ${QDRANT_URL}"
        printf '       existing collections: %s\n' "$existing"
        printf '       run: python -m src.index.upsert\n'
        exit 2
        ;;
    ERROR*)
        msg=$(printf '%s' "$qdrant_check_output" | cut -f2)
        fail "Qdrant unreachable: ${msg}"
        exit 2
        ;;
    *)
        fail "unexpected probe output: ${qdrant_check_output}"
        exit 2
        ;;
esac

# ---- Step 3: smoke queries -------------------------------------------
step "Running 5-query smoke test"

if python -m scripts.smoke_prod; then
    ok "all smoke queries passed"
    printf '\n%sDeploy looks healthy. Ready to publish the URL.%s\n' "${GREEN}${BOLD}" "${RESET}"
    exit 0
else
    smoke_exit=$?
    fail "smoke test exited with code ${smoke_exit}"
    exit "$smoke_exit"
fi
