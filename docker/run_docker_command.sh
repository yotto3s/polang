#!/bin/bash

set -eu
set -o pipefail

SCRIPT_DIR=$(dirname "$0")

. "${SCRIPT_DIR}/docker_config.sh"

# Check if the container is running
if [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null)" = "true" ]; then
    # Execute the remaining arguments as a command inside the container
    docker exec -it "${CONTAINER_NAME}" "$@"
else
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    exit 1
fi
