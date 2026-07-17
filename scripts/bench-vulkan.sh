#!/usr/bin/env bash
# Vulkan benchmark driver: runs tp-bench under the Vulkan runtime env with
# nvidia-smi VRAM sampling (200 ms). Usage:
#   scripts/bench-vulkan.sh <tag> <ftype> [env VAR=1 ...]
# Produces bench-vk-<tag>.json and prints the sampled peak VRAM.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=$1; FTYPE=$2; shift 2

VKLOADER=$(nix build nixpkgs#vulkan-loader --print-out-paths --no-link)
export LD_LIBRARY_PATH="$VKLOADER/lib:/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

HARNESS=/home/rich/c/float-harness
VRAMLOG=$(mktemp /tmp/vram-$TAG.XXXX)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$VRAMLOG" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

# TP_BENCH_ARGS: extra tp-bench arguments (e.g. "--rows mdec,pipe2")
env "$@" ./build/vulkan/tp-bench --gguf "$HARNESS/ggufs" --ref "$HARNESS/dumps" \
    --device gpu --ftype "$FTYPE" --json "bench-vk-$TAG.json" ${TP_BENCH_ARGS:-} </dev/null

kill $SMI 2>/dev/null || true; wait $SMI 2>/dev/null || true
PEAK=$(sort -n "$VRAMLOG" | tail -1)
BASE=$(head -1 "$VRAMLOG")
echo "vram_sampled_peak_mb=$PEAK vram_first_sample_mb=$BASE samples=$(wc -l < "$VRAMLOG")"
