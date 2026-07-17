#!/usr/bin/env python3
"""Assemble bench-vulkan.json from the per-config bench-vk-*.json runs.

Top-level shape mirrors bench.json (label/rows/peak_vram_mb = the headline
fully-on-GPU f16 fast-graph config); the other measured configurations are
kept under "configs" with their sampled-VRAM peaks patched in from the run
log (scripts/bench-vulkan.sh prints vram_sampled_peak_mb per run).
"""
import json
import sys

# (tag, config description, sampled peak MiB from vk-final-runs.log /
#  bench-vulkan-runs.log)
CONFIGS = [
    ("fast-f16-gpu", "fast graph, custom ops on Vulkan, default shader math", None),
    ("fast-f32-gpu", "fast graph f32, custom ops on Vulkan, default shader math", None),
    ("exact-f16-gpu", "conservative graph (TP_MDEC_EXACT=1), custom ops on Vulkan, default math", None),
    ("fast-f16-fallback", "BASELINE fast graph, custom ops on CPU (TP_FORCE_FALLBACK=1); mdec+pipe2 rows only", None),
    ("fast-f16-strict", "fast graph, scalar-f32 shader math (GGML_VK_DISABLE_F16/COOPMAT/COOPMAT2); no pipeline rows", None),
    ("exact-f16", "BASELINE conservative graph, bilinear_gather on CPU (pre-kernel)", None),
    ("exact-f32", "BASELINE conservative graph f32, bilinear_gather on CPU (pre-kernel)", None),
]

def main(out_path):
    # header substring -> config tag (explicit, most-specific matching)
    HEADERS = [
        ("force-fallback", "fast-f16-fallback"),
        ("strict", "fast-f16-strict"),
        ("bench vk-fast-f16", "fast-f16-gpu"),
        ("bench vk-fast-f32", "fast-f32-gpu"),
        ("bench vk-exact-f16", "exact-f16-gpu"),
        ("== exact-f16 (default math)", "exact-f16"),
        ("== exact-f32 (default math)", "exact-f32"),
    ]
    sampled = {}
    for log in ("vk-final-runs.log", "bench-vulkan-runs.log"):
        try:
            tag = None
            for line in open(log):
                if line.startswith("====") or line.startswith("== "):
                    tag = next((t for h, t in HEADERS if h in line), None)
                elif line.startswith("vram_sampled_peak_mb=") and tag:
                    sampled[tag] = int(line.split("=")[1].split()[0])
                    tag = None
        except FileNotFoundError:
            pass

    configs = []
    for tag, desc, _ in CONFIGS:
        try:
            d = json.load(open(f"bench-vk-{tag}.json"))
        except FileNotFoundError:
            continue
        d["config"] = desc
        d["label"] = f"cpp-vulkan-{tag}"
        if tag in sampled:
            d["peak_vram_sampled_mb"] = sampled[tag]
        configs.append(d)

    head = next(c for c in configs if c["label"].endswith("fast-f16-gpu"))
    out = {
        "label": "cpp-f16-vulkan",
        "rows": head["rows"],
        "peak_vram_mb": head["peak_vram_mb"],
        "peak_vram_sampled_mb": head.get("peak_vram_sampled_mb"),
        "configs": configs,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {out_path} with {len(configs)} configs")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench-vulkan.json")
