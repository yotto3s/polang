# Two-Stage Docker Image Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the single Dockerfile into a base image (CI) and dev image (local development with tools + Claude Code).

**Architecture:** Base image (`Dockerfile.base`) contains only build/lint dependencies and is published to ghcr.io. Dev image (`Dockerfile.dev`) extends base with developer utilities, Claude Code, and a gosu-based entrypoint for UID/GID remapping. Chezmoi is included for optional dotfiles management.

**Tech Stack:** Docker, GitHub Actions, gosu, chezmoi, nodesource Node.js 22

**Design doc:** `docs/plans/2026-02-14-docker-two-stage-design.md`

---

### Task 1: Create Dockerfile.base

**Files:**
- Create: `docker/Dockerfile.base`

**Step 1: Create Dockerfile.base from existing Dockerfile**

Copy the existing `docker/Dockerfile` to `docker/Dockerfile.base` with these changes:
- Remove the `ARG USERNAME`, `ARG USER_UID`, `ARG USER_GID` build args
- Hardcode `devuser` with UID/GID 1000
- Remove `USER $USERNAME` at the end (base runs as root in CI)
- Keep everything else identical

```dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install essential development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    flex \
    bison \
    gcc \
    g++ \
    gdb \
    git \
    sudo \
    lsb-release \
    software-properties-common \
    gnupg \
    libzstd-dev \
    libz-dev \
    lcov \
    python3 \
    && apt-get clean

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo ./llvm.sh 20 all && \
    rm llvm.sh

# Install MLIR development packages
RUN apt-get update && apt-get install -y \
    libmlir-20-dev \
    mlir-20-tools \
    && apt-get clean

# Configure LLVM library path for dynamic linker
RUN echo "/usr/lib/llvm-20/lib" > /etc/ld.so.conf.d/llvm-20.conf && ldconfig

# Register clang alternatives
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100 \
    --slave /usr/bin/clang++ clang++ /usr/bin/clang++-20 \
    --slave /usr/bin/clang-format clang-format /usr/bin/clang-format-20 \
    --slave /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-20 \
    --slave /usr/bin/clangd clangd /usr/bin/clangd-20

# Remove default ubuntu user
RUN touch /var/mail/ubuntu && chown ubuntu /var/mail/ubuntu && userdel -r ubuntu

# Create default devuser (UID/GID 1000) with sudo privileges
RUN groupadd --gid 1000 devuser \
    && useradd --uid 1000 --gid 1000 -m devuser \
    && echo "devuser ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/devuser \
    && chmod 0440 /etc/sudoers.d/devuser

# Set the working directory
WORKDIR /workspace
RUN chown 1000:1000 /workspace

CMD ["/bin/bash"]
```

**Step 2: Verify Dockerfile.base builds**

Run: `docker build -f docker/Dockerfile.base -t polang-base-test docker/`
Expected: Builds successfully (this will take a while due to LLVM)

**Step 3: Verify base image can build the project**

Run:
```bash
docker run --rm -v "$(pwd):/workspace/polang" -w /workspace/polang --user root polang-base-test \
    bash -c "cmake --preset clang-debug && cmake --build --preset clang-debug -j$(nproc) && ctest --preset clang-debug"
```
Expected: Configure, build, and tests all pass

**Step 4: Commit**

```bash
git add docker/Dockerfile.base
git commit -m "docker: add Dockerfile.base for CI builds"
```

---

### Task 2: Create entrypoint.sh

**Files:**
- Create: `docker/entrypoint.sh`

**Step 1: Create the entrypoint script**

```bash
#!/bin/bash
set -e

TARGET_UID=${TARGET_UID:-1000}
TARGET_GID=${TARGET_GID:-1000}
USERNAME=devuser

# Adjust GID if different from default
if [ "$(id -g "$USERNAME")" != "$TARGET_GID" ]; then
    groupmod -g "$TARGET_GID" "$USERNAME"
fi

# Adjust UID if different from default
if [ "$(id -u "$USERNAME")" != "$TARGET_UID" ]; then
    usermod -u "$TARGET_UID" -o "$USERNAME"
fi

# Fix home directory ownership
chown -R "$TARGET_UID:$TARGET_GID" "/home/$USERNAME"

# Apply chezmoi dotfiles if repo is specified
if [ -n "$CHEZMOI_REPO" ]; then
    gosu "$USERNAME" chezmoi init --apply "$CHEZMOI_REPO"
fi

# Drop to user and exec the command
exec gosu "$USERNAME" "$@"
```

**Step 2: Make it executable and commit**

```bash
chmod +x docker/entrypoint.sh
git add docker/entrypoint.sh
git commit -m "docker: add entrypoint.sh for UID/GID remapping"
```

---

### Task 3: Create Dockerfile.dev

**Files:**
- Create: `docker/Dockerfile.dev`

**Step 1: Create the dev Dockerfile**

```dockerfile
FROM ghcr.io/yotto3s/polang-base:latest

ENV DEBIAN_FRONTEND=noninteractive

# Entrypoint dependency
RUN apt-get update && apt-get install -y \
    gosu \
    && apt-get clean

# Editors
RUN apt-get update && apt-get install -y \
    neovim \
    python3-pynvim \
    && apt-get clean

# Search and navigation
RUN apt-get update && apt-get install -y \
    ripgrep \
    fd-find \
    fzf \
    && apt-get clean
RUN ln -s /usr/bin/fdfind /usr/bin/fd

# File viewers
RUN apt-get update && apt-get install -y \
    bat \
    exa \
    && apt-get clean
RUN ln -s /usr/bin/batcat /usr/bin/bat 2>/dev/null || true

# VCS tools
RUN apt-get update && apt-get install -y \
    tig \
    && apt-get clean

# System monitoring
RUN apt-get update && apt-get install -y \
    htop \
    && apt-get clean

# Debugging and profiling
RUN apt-get update && apt-get install -y \
    valgrind \
    strace \
    linux-tools-generic \
    && apt-get clean

# Utilities
RUN apt-get update && apt-get install -y \
    tmux \
    parallel \
    curl \
    tree \
    unzip \
    && apt-get clean

# Tools installed from binary releases (not in Ubuntu repos)
# lazygit
RUN LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*') \
    && curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz" \
    && tar xf lazygit.tar.gz lazygit \
    && install lazygit /usr/local/bin \
    && rm lazygit lazygit.tar.gz

# zoxide
RUN curl -sSfL https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | bash

# starship
RUN curl -sS https://starship.rs/install.sh | sh -s -- -y

# dust
RUN DUST_VERSION=$(curl -s "https://api.github.com/repos/bootandy/dust/releases/latest" | grep -Po '"tag_name": "v\K[^"]*') \
    && curl -Lo dust.tar.gz "https://github.com/bootandy/dust/releases/latest/download/dust-v${DUST_VERSION}-x86_64-unknown-linux-gnu.tar.gz" \
    && tar xf dust.tar.gz --strip-components=1 -C /usr/local/bin \
    && rm dust.tar.gz

# duf
RUN DUF_VERSION=$(curl -s "https://api.github.com/repos/muesli/duf/releases/latest" | grep -Po '"tag_name": "v\K[^"]*') \
    && curl -Lo duf.deb "https://github.com/muesli/duf/releases/latest/download/duf_${DUF_VERSION}_linux_amd64.deb" \
    && dpkg -i duf.deb \
    && rm duf.deb

# procs
RUN PROCS_VERSION=$(curl -s "https://api.github.com/repos/dalance/procs/releases/latest" | grep -Po '"tag_name": "v\K[^"]*') \
    && curl -Lo procs.zip "https://github.com/dalance/procs/releases/latest/download/procs-v${PROCS_VERSION}-x86_64-linux.zip" \
    && unzip procs.zip \
    && install procs /usr/local/bin \
    && rm procs procs.zip

# bottom
RUN BOTTOM_VERSION=$(curl -s "https://api.github.com/repos/ClementTsang/bottom/releases/latest" | grep -Po '"tag_name": "\K[^"]*') \
    && curl -Lo bottom.deb "https://github.com/ClementTsang/bottom/releases/latest/download/bottom_${BOTTOM_VERSION}-1_amd64.deb" \
    && dpkg -i bottom.deb \
    && rm bottom.deb

# Chezmoi (for dotfiles management)
RUN sh -c "$(curl -fsLS get.chezmoi.io)" -- -b /usr/local/bin

# Node.js 22 + Claude Code
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && apt-get clean

# Entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
CMD ["/bin/bash"]
```

**Step 2: Build dev image (using local base for testing)**

Run: `docker build -f docker/Dockerfile.dev --build-arg BASE_IMAGE=polang-base-test -t polang-dev-test docker/`

Note: For the first local test, temporarily change the FROM line to `polang-base-test` since `ghcr.io/yotto3s/polang-base:latest` doesn't exist yet. Revert before committing.

Actually, since the base image isn't published yet, use a build arg:

Change the first line of Dockerfile.dev to:
```dockerfile
ARG BASE_IMAGE=ghcr.io/yotto3s/polang-base:latest
FROM ${BASE_IMAGE}
```

Then test with:
```bash
docker build -f docker/Dockerfile.dev --build-arg BASE_IMAGE=polang-base-test -t polang-dev-test docker/
```

Expected: Builds successfully

**Step 3: Verify dev image works with entrypoint**

Run:
```bash
docker run --rm \
    -e TARGET_UID=$(id -u) \
    -e TARGET_GID=$(id -g) \
    polang-dev-test id
```
Expected: Shows your host UID/GID

Run:
```bash
docker run --rm \
    -e TARGET_UID=$(id -u) \
    -e TARGET_GID=$(id -g) \
    polang-dev-test bash -c "which claude && which nvim && which rg && which lazygit"
```
Expected: All tool paths printed

**Step 4: Verify dev image can build the project**

Run:
```bash
docker run --rm \
    -v "$(pwd):/workspace/polang" \
    -w /workspace/polang \
    -e TARGET_UID=$(id -u) \
    -e TARGET_GID=$(id -g) \
    polang-dev-test \
    bash -c "cmake --preset clang-debug && cmake --build --preset clang-debug -j$(nproc) && ctest --preset clang-debug"
```
Expected: Configure, build, and tests all pass

**Step 5: Commit**

```bash
git add docker/Dockerfile.dev
git commit -m "docker: add Dockerfile.dev with dev tools and Claude Code"
```

---

### Task 4: Update docker scripts

**Files:**
- Modify: `docker/docker_config.sh`
- Modify: `docker/docker_build.sh`
- Modify: `docker/docker_run.sh`

**Step 1: Update docker_config.sh**

Replace contents with:

```bash
export BASE_IMAGE=ghcr.io/yotto3s/polang-base:latest
export DEV_IMAGE=polang-dev
export CONTAINER_NAME=polang
export ROOT_DIR=/workspace/polang
export BUILD_DIR=${ROOT_DIR}/build
```

**Step 2: Update docker_build.sh**

Replace contents with:

```bash
#!/usr/bin/bash

set -eux
set -o pipefail

SCRIPT_DIR=$(dirname "$0")

. "${SCRIPT_DIR}/docker_config.sh"

docker pull "${BASE_IMAGE}"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile.dev" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${DEV_IMAGE}" \
    "${SCRIPT_DIR}"
```

**Step 3: Update docker_run.sh**

Replace contents with:

```bash
#!/usr/bin/bash

set -eux
set -o pipefail

SCRIPT_FULL_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_FULL_PATH}")
PROJECT_DIR="${SCRIPT_DIR}/../"

. "${SCRIPT_DIR}/docker_config.sh"

docker run -d \
    --name "${CONTAINER_NAME}" \
    -v "${PROJECT_DIR}:/workspace/polang" \
    -v "${HOME}/.claude:/home/devuser/.claude" \
    -e TARGET_UID="$(id -u)" \
    -e TARGET_GID="$(id -g)" \
    -w /workspace/polang \
    "${DEV_IMAGE}"
```

**Step 4: Test the full local workflow**

Run:
```bash
# Clean up any existing container
docker rm -f polang 2>/dev/null || true

# Build (uses local base for now)
# Temporarily edit docker_config.sh to use BASE_IMAGE=polang-base-test
docker/docker_build.sh

# Run
docker/docker_run.sh

# Verify
docker exec polang id
docker exec polang bash -c "cd /workspace/polang && cmake --preset clang-debug && cmake --build --preset clang-debug -j\$(nproc) && ctest --preset clang-debug"
```

Expected: Container starts, UID matches host, project builds and tests pass

**Step 5: Commit**

```bash
git add docker/docker_config.sh docker/docker_build.sh docker/docker_run.sh
git commit -m "docker: update scripts for two-stage image workflow"
```

---

### Task 5: Delete old Dockerfile

**Files:**
- Delete: `docker/Dockerfile`

**Step 1: Remove the old Dockerfile**

```bash
git rm docker/Dockerfile
git commit -m "docker: remove old single-stage Dockerfile"
```

---

### Task 6: Update docker.yml workflow

**Files:**
- Modify: `.github/workflows/docker.yml`

**Step 1: Update the workflow**

Replace contents with:

```yaml
name: Docker

on:
  push:
    branches: [main]
    paths:
      - 'docker/Dockerfile.base'
      - '.github/workflows/docker.yml'
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository_owner }}/polang-base

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: ./docker
          file: ./docker/Dockerfile.base
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

**Step 2: Commit**

```bash
git add .github/workflows/docker.yml
git commit -m "ci: update docker.yml to build polang-base image"
```

---

### Task 7: Update ci.yml workflow

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Update the workflow**

Changes required:
1. `IMAGE_NAME` env var: `${{ github.repository_owner }}/polang-base` (was `${{ github.repository }}-dev`)
2. `check-changes` path filter: watch `docker/Dockerfile.base` instead of `docker/**`
3. `build-image` job: use `file: ./docker/Dockerfile.base` in build-push-action
4. All `container.image` references: `ghcr.io/${{ github.repository_owner }}/polang-base:latest`
5. `cache-from` reference: update to match new image name

Replace the full file with:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository_owner }}/polang-base

jobs:
  check-changes:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    outputs:
      docker-changed: ${{ steps.filter.outputs.docker }}
    name: Check Changes

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Check for Docker changes
        uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            docker:
              - 'docker/Dockerfile.base'

  build-image:
    needs: [check-changes]
    if: needs.check-changes.outputs.docker-changed == 'true'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    name: Build Docker Image

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: ./docker
          file: ./docker/Dockerfile.base
          push: ${{ github.event_name != 'pull_request' }}
          load: ${{ github.event_name == 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-to: type=inline

  format-check:
    runs-on: ubuntu-24.04
    container:
      image: ghcr.io/${{ github.repository_owner }}/polang-base:latest
      credentials:
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
      options: --user root
    name: Format Check

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Check formatting
        run: ./scripts/run-clang-format.sh --check

  lint:
    runs-on: ubuntu-24.04
    container:
      image: ghcr.io/${{ github.repository_owner }}/polang-base:latest
      credentials:
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
      options: --user root
    name: Clang-Tidy

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure
        run: cmake --preset lint

      - name: Build
        run: cmake --build --preset lint

      - name: Run clang-tidy
        run: ./scripts/run-clang-tidy.sh build/lint

  build-and-test:
    needs: [format-check, lint, build-image]
    if: >-
      always() &&
      needs.format-check.result == 'success' &&
      needs.lint.result == 'success' &&
      (needs.build-image.result == 'success' || needs.build-image.result == 'skipped')
    runs-on: ubuntu-24.04
    container:
      image: ghcr.io/${{ github.repository_owner }}/polang-base:latest
      credentials:
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
      options: --user root
    strategy:
      matrix:
        preset: [gcc-debug, gcc-release, clang-debug, clang-release]

    name: Build (${{ matrix.preset }})

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure
        run: cmake --preset ${{ matrix.preset }}

      - name: Build
        run: cmake --build --preset ${{ matrix.preset }}

      - name: Run unit tests
        run: ctest --preset ${{ matrix.preset }}

      - name: Run example programs
        run: |
          for f in example/*.po; do
            echo "=== $(basename $f) ==="
            ./build/${{ matrix.preset }}/bin/PolangRepl "$f"
          done

  sanitizers:
    needs: [build-and-test]
    if: ${{ !cancelled() && needs.build-and-test.result == 'success' }}
    runs-on: ubuntu-24.04
    container:
      image: ghcr.io/${{ github.repository_owner }}/polang-base:latest
      credentials:
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
      options: --user root
    strategy:
      matrix:
        preset: [asan, ubsan]

    name: Sanitizer (${{ matrix.preset }})

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure
        run: cmake --preset ${{ matrix.preset }}

      - name: Build
        run: cmake --build --preset ${{ matrix.preset }}

      - name: Run unit tests
        run: ctest --preset ${{ matrix.preset }}

      - name: Run example programs
        env:
          ASAN_OPTIONS: detect_leaks=1:halt_on_error=1
          UBSAN_OPTIONS: halt_on_error=1:print_stacktrace=1
        run: |
          for f in example/*.po; do
            echo "=== $(basename $f) ==="
            ./build/${{ matrix.preset }}/bin/PolangRepl "$f"
          done

  coverage:
    needs: [build-and-test]
    if: ${{ !cancelled() && needs.build-and-test.result == 'success' }}
    runs-on: ubuntu-24.04
    container:
      image: ghcr.io/${{ github.repository_owner }}/polang-base:latest
      credentials:
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
      options: --user root
    name: Code Coverage

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure
        run: cmake --preset coverage

      - name: Build
        run: cmake --build --preset coverage

      - name: Run tests
        run: ctest --preset coverage

      - name: Generate coverage report
        run: cmake --build build/coverage --target coverage

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: build/coverage/coverage.info
          fail_ci_if_error: true
          verbose: true
        env:
          CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
```

**Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: update ci.yml to use polang-base image"
```

---

### Task 8: Update documentation

**Files:**
- Modify: `doc/Building.md` (Docker Environment section)
- Modify: `CLAUDE.md` (Docker Environment section)

**Step 1: Update doc/Building.md**

In the "Docker Environment (Recommended)" section (~lines 18-36), update to describe the two-stage setup:

Replace the current Docker section with:

```markdown
### Docker Environment (Recommended)

The project uses a two-stage Docker setup:

**Base image** (`ghcr.io/yotto3s/polang-base`) — used by CI:
- Ubuntu 24.04 base
- GCC and Clang 20 compilers
- CMake, Bison, Flex
- LLVM 20 with MLIR
- clang-format, clang-tidy, clangd
- lcov (for coverage), Python 3 (for lit tests)

**Dev image** (`polang-dev`) — built locally for development:
- Everything in base, plus:
- neovim, ripgrep, fd, fzf, bat, lazygit, tmux, starship, etc.
- valgrind, strace, perf (debugging/profiling)
- Claude Code (via Node.js)
- chezmoi (optional dotfiles management)
- gosu-based UID/GID remapping

\`\`\`bash
# Build the dev image locally (pulls base from ghcr.io)
docker/docker_build.sh

# Start a container
docker/docker_run.sh

# Run any command inside the docker container
docker exec polang <command> [options]
\`\`\`
```

**Step 2: Update CLAUDE.md**

In the "Docker Environment" section, update the docker_build.sh comment:

Replace:
```markdown
# Build the Docker image locally
docker/docker_build.sh
```

With:
```markdown
# Build the dev image locally (pulls base from ghcr.io)
docker/docker_build.sh
```

**Step 3: Commit**

```bash
git add doc/Building.md CLAUDE.md
git commit -m "docs: update Docker documentation for two-stage setup"
```

---

### Task 9: Bootstrap — push base image to ghcr.io

This task requires manual intervention since the base image must exist on ghcr.io before the dev image can be built normally.

**Step 1: Push branch and trigger docker.yml**

The `docker.yml` workflow only runs on `main`. Options:
- Merge to main and let the workflow run
- Or manually trigger via `workflow_dispatch` after merge

**Step 2: Verify the base image is available**

Run: `docker pull ghcr.io/yotto3s/polang-base:latest`
Expected: Image pulls successfully

**Step 3: Verify dev image builds from published base**

Run:
```bash
docker rm -f polang 2>/dev/null || true
docker/docker_build.sh
docker/docker_run.sh
docker exec polang id
docker exec polang bash -c "cd /workspace/polang && cmake --preset clang-debug && cmake --build --preset clang-debug -j\$(nproc) && ctest --preset clang-debug"
```
Expected: Everything works end-to-end

**Step 4: Verify CI passes**

Push a PR and confirm all CI jobs use the base image and pass.
