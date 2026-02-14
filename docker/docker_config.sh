export BASE_IMAGE=ghcr.io/yotto3s/polang-base:latest
export DEV_IMAGE=polang-dev
export CONTAINER_NAME=polang
export ROOT_DIR=/workspace/polang
export BUILD_DIR=${ROOT_DIR}/build

# Claude Code authentication (picks the first available method):
#   - ANTHROPIC_API_KEY: for API pay-per-use plans
#   - CLAUDE_CODE_OAUTH_TOKEN: for subscription plans (Pro/Max) — auto-extracted below
CREDS_FILE="${HOME}/.claude/.credentials.json"
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -f "$CREDS_FILE" ]; then
    CLAUDE_CODE_OAUTH_TOKEN=$(python3 -c "import json; print(json.load(open('$CREDS_FILE'))['claudeAiOauth']['accessToken'])" 2>/dev/null) || true
    export CLAUDE_CODE_OAUTH_TOKEN
fi

# Dotfiles: set to a chezmoi-compatible repo URL to apply on container start
# export CHEZMOI_REPO=https://github.com/username/dotfiles.git
# export CHEZMOI_EMAIL=you@example.com
