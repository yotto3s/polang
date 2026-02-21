#!/usr/bin/bash

set -eu
set -o pipefail

SCRIPT_FULL_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_FULL_PATH}")
PROJECT_DIR="${SCRIPT_DIR}/../"

. "${SCRIPT_DIR}/docker_config.sh"

# Mount host .gitconfig if it exists (read-only for git identity inheritance)
GITCONFIG_MOUNT=()
if [ -f "${HOME}/.gitconfig" ]; then
    GITCONFIG_MOUNT=(-v "${HOME}/.gitconfig:/home/devuser/.gitconfig:ro")
fi

docker run -d \
    --name "${CONTAINER_NAME}" \
    -v "${PROJECT_DIR}:/workspace/polang" \
    -v "${HOME}/.claude:/home/devuser/.claude" \
    -v "${HOME}/.ssh:/home/devuser/.ssh:ro" \
    ${GITCONFIG_MOUNT[@]+"${GITCONFIG_MOUNT[@]}"} \
    -e TARGET_UID="$(id -u)" \
    -e TARGET_GID="$(id -g)" \
    ${ANTHROPIC_API_KEY:+-e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"} \
    ${CLAUDE_CODE_OAUTH_TOKEN:+-e CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN"} \
    -w /workspace/polang \
    "${DEV_IMAGE}" \
    sleep infinity
