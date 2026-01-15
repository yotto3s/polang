#!/usr/bin/bash

set -eux
set -o pipefail

SCRIPT_DIR=$(dirname "$0")

. "${SCRIPT_DIR}/docker_config.sh"

docker build \
    --build-arg USERNAME="$(id -un)" \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    -t "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"
