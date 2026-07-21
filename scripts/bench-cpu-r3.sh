#!/usr/bin/env bash
# R3 bench contract: write bench-cpu-r3.json (tp-bench --json format) at the
# repo root. Copy to scripts/bench-cpu-r3.sh and adapt (build dir, --ftype,
# TP_*/GGML_* env, thread count) as you optimize — the gate runs this script
# verbatim and parses the JSON it writes.
set -e
cd "$(dirname "$0")/.."
H=/home/rich/c/float-harness
# R3 CPU ship config: clean release build (GGML_NATIVE=ON -> AVX512), 12
# threads (physical cores; SMT hurts the AVX512 GEMMs), fused custom ops
# (faster than stock ggml once the kernels are vectorized). See docs/r3-summary.md.
: "${TP_NTHREADS:=12}"
export TP_NTHREADS
exec ./build/release/tp-bench --gguf "$H/ggufs" --ref "$H/dumps" \
  --device cpu --ftype "${TP_FTYPE:-f32}" --json bench-cpu-r3.json
