#!/usr/bin/bash

set -eu
set -o pipefail

SCRIPT_FULL_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_FULL_PATH}")
PROJECT_DIR="${SCRIPT_DIR}/../"

. "${SCRIPT_DIR}/docker_config.sh"

docker run -d \
    --name "${CONTAINER_NAME}" \
    -v "${PROJECT_DIR}:/workspace/polang" \
    -v "${HOME}/.claude:/home/devuser/.claude" \
    -v "${HOME}/.ssh:/home/devuser/.ssh:ro" \
    -e TARGET_UID="$(id -u)" \
    -e TARGET_GID="$(id -g)" \
    ${ANTHROPIC_API_KEY:+-e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"} \
    ${CLAUDE_CODE_OAUTH_TOKEN:+-e CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN"} \
    ${CHEZMOI_REPO:+-e CHEZMOI_REPO="$CHEZMOI_REPO"} \
    ${CHEZMOI_EMAIL:+-e CHEZMOI_EMAIL="$CHEZMOI_EMAIL"} \
    -w /workspace/polang \
    "${DEV_IMAGE}" \
    sleep infinity
