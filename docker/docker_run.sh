#!/usr/bin/bash

set -eux
set -o pipefail

SCRIPT_FULL_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_FULL_PATH}")
PROJECT_DIR="${SCRIPT_DIR}/../"

. "${SCRIPT_DIR}/docker_config.sh"

docker run -d --name "${CONTAINER_NAME}" -v "${PROJECT_DIR}:/workspace/polang" -w "/workspace/polang" polang-dev sleep infinity
