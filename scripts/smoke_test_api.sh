#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen3-30B-A3B-Instruct-2507}"

curl --fail --silent --show-error "$BASE_URL/v1/models" | python -m json.tool
curl --fail --silent --show-error "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":8}" | python -m json.tool
curl --fail --no-buffer --silent --show-error "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to three.\"}],\"max_tokens\":16,\"stream\":true}" | sed -n '1,4p'
