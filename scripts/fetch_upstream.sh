#!/usr/bin/env bash
# Clone (or update) the public problem definitions used by tools/cuemu/run_tests.py.
# They are NOT vendored into this repo: LeetGPU's challenges are CC BY-NC-ND and
# Tensara's problem repo has no license, so we only reference them.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .upstream

clone() {
    local url=$1 dir=$2
    if [ -d "$dir/.git" ]; then
        git -C "$dir" pull --ff-only --quiet
    else
        GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --quiet "$url" "$dir"
    fi
    echo "$dir @ $(git -C "$dir" rev-parse --short HEAD)"
}

clone https://github.com/AlphaGPU/leetgpu-challenges .upstream/leetgpu-challenges
clone https://github.com/tensara/problems .upstream/tensara-problems
clone https://github.com/tensara/tensara .upstream/tensara
