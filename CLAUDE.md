# talking-portrait.cpp

C++/GGML port of FLOAT. The reference harness lives at
/home/rich/c/float-harness (read-only; see its README and docs/).

- Build: CMake ≥ 3.24, presets preferred (debug = ASan/UBSan, release,
  fuzz = clang + libFuzzer). CUDA arch: 120 (RTX 5070 Ti).
- Harness commands run via /home/rich/c/float-harness/bin/refrun (python) and
  /home/rich/c/float-harness/bin/gate (milestone gates).
- Parity dumps: ./parity-dumps/<case>/ ; generated frames: ./videos/.
- NixOS host: NVIDIA driver libs at /run/opengl-driver/lib.
