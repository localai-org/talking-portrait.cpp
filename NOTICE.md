# License notice

Code in this repository (everything outside `ggml/` and `third_party/`) is
MIT — see `LICENSE`.

## Vendored code

- `ggml/` (submodule, plus `patches/ggml-tp-ops.patch`) is MIT, copyright
  the ggml authors.
- `third_party/stb/` is dual-licensed MIT / public domain (see the end of
  each header).

## FLOAT weights are non-commercial — do not redistribute anything derived from them

This is a port of [FLOAT](https://github.com/deepbrainai-research/float),
whose official weight release is CC BY-NC 4.0 (its README states
CC BY-NC-ND 4.0). Non-commercial use is certain under every reading;
whether a *derivative* (converted GGUFs, parity/activation dumps, possibly
generated video) may be redistributed is ambiguous.

Consequence, enforced by `.gitignore`: GGUFs, `parity-dumps*/`, `videos*/`
and other weight-derived artifacts are generated locally on each machine and
are **never committed, published, or uploaded anywhere**.

The wav2vec2 backbones (`facebook/wav2vec2-base-960h`,
`r-f/wav2vec-english-speech-emotion-recognition`) are Apache-2.0 and carry no
such restriction, but all model artifacts are treated uniformly.

Commercial licensing contact per upstream README: daniel@deepbrain.io.
