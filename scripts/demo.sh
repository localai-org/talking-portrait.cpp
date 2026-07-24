#!/usr/bin/env bash
# Build and launch the local CUDA/WebGL demo.
#
#   scripts/demo.sh
#   PORT=9000 GGUF_DIR=/path/to/ggufs STORE_DIR=/path/to/results scripts/demo.sh
set -euo pipefail

TP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP_PORT="${PORT:-4001}"
TP_GGUF_DIR="${GGUF_DIR:-/home/rich/c/float-harness/ggufs}"
TP_STORE_DIR="${STORE_DIR:-$TP_ROOT/generations}"
TP_NETWORK="talking-portrait-demo"
TP_APP_CONTAINER="talking-portrait-demo-run"
TP_CADDY_CONTAINER="talking-portrait-caddy"
TP_CERT_SANS="DNS:localhost,DNS:$(hostname)"
for TP_HOST_IP in $(hostname -I); do
  if [[ "$TP_HOST_IP" != *:* ]]; then
    TP_CERT_SANS+=",IP:$TP_HOST_IP"
  fi
done
TP_IMAGE="${DEMO_IMAGE:-talking-portrait-demo}"

if [[ ! -d "$TP_GGUF_DIR" ]]; then
  echo "GGUF directory not found: $TP_GGUF_DIR" >&2
  echo "Set GGUF_DIR to the directory containing audio_f16.gguf, fmt_f16.gguf, and motion_ae_f16.gguf." >&2
  exit 2
fi
mkdir -p "$TP_STORE_DIR"

if ! docker image inspect "$TP_IMAGE" >/dev/null 2>&1; then
  if [[ -z "${DEMO_IMAGE:-}" ]] &&
     docker image inspect trellis2-demo >/dev/null 2>&1; then
    docker build -f "$TP_ROOT/docker/Dockerfile.reuse" \
      --build-arg BASE_IMAGE=trellis2-demo -t "$TP_IMAGE" "$TP_ROOT"
  else
    docker build -f "$TP_ROOT/docker/Dockerfile.demo" -t "$TP_IMAGE" "$TP_ROOT"
  fi
fi
if ! docker run --rm "$TP_IMAGE" sh -lc \
  'command -v ffmpeg >/dev/null && command -v openssl >/dev/null'; then
  echo "Demo image $TP_IMAGE must provide ffmpeg and openssl." >&2
  exit 2
fi

docker run --rm --device nvidia.com/gpu=all \
  -v "$TP_ROOT:/work" -w /work \
  -e GOCACHE=/tmp/go-build -e CGO_ENABLED=0 \
  "$TP_IMAGE" bash -lc '
    cmake -B build-cuda-shared -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTP_SHARED=ON -DTP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DTP_BUILD_TESTS=OFF -DTP_BUILD_TOOLS=OFF &&
    cmake --build build-cuda-shared -j"$(nproc)" &&
    cd server && go build -o talking-portrait-server-linux .
  '

docker network inspect "$TP_NETWORK" >/dev/null 2>&1 ||
  docker network create "$TP_NETWORK" >/dev/null
docker rm -f "$TP_CADDY_CONTAINER" "$TP_APP_CONTAINER" >/dev/null 2>&1 || true

# Generate a persistent local CA once, then refresh the one-year server
# certificate on every launch so changed LAN/Tailscale addresses are covered.
docker run --rm -v talking_portrait_caddy_data:/data \
  -e "TP_CERT_SANS=$TP_CERT_SANS" "$TP_IMAGE" bash -lc '
    set -euo pipefail
    TP_CERT_DIR=/data/talking-portrait
    mkdir -p "$TP_CERT_DIR"
    if [[ ! -s "$TP_CERT_DIR/root.crt" || ! -s "$TP_CERT_DIR/root.key" ]]; then
      openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 \
        -out "$TP_CERT_DIR/root.key"
      openssl req -x509 -new -key "$TP_CERT_DIR/root.key" -sha256 -days 3650 \
        -subj "/CN=Talking Portrait Local CA" -out "$TP_CERT_DIR/root.crt" \
        -addext "basicConstraints=critical,CA:TRUE" \
        -addext "keyUsage=critical,keyCertSign,cRLSign"
    fi
    openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 \
      -out "$TP_CERT_DIR/server.key"
    openssl req -new -key "$TP_CERT_DIR/server.key" \
      -subj "/CN=talking-portrait" -out "$TP_CERT_DIR/server.csr"
    {
      echo "basicConstraints=critical,CA:FALSE"
      echo "keyUsage=critical,digitalSignature,keyEncipherment"
      echo "extendedKeyUsage=serverAuth"
      echo "subjectAltName=$TP_CERT_SANS"
    } > "$TP_CERT_DIR/server.ext"
    openssl x509 -req -in "$TP_CERT_DIR/server.csr" \
      -CA "$TP_CERT_DIR/root.crt" -CAkey "$TP_CERT_DIR/root.key" \
      -CAcreateserial -sha256 -days 365 -extfile "$TP_CERT_DIR/server.ext" \
      -out "$TP_CERT_DIR/server.crt"
    rm -f "$TP_CERT_DIR/server.csr" "$TP_CERT_DIR/server.ext"
  '

docker run -d --rm --name "$TP_APP_CONTAINER" --network "$TP_NETWORK" \
  --device nvidia.com/gpu=all \
  -v "$TP_ROOT:/work" -v "$TP_GGUF_DIR:/models:ro" -v "$TP_STORE_DIR:/generations" \
  -w /work/server "$TP_IMAGE" \
  ./talking-portrait-server-linux -lib /work/build-cuda-shared/libtp.so \
  -ggufs /models -device cuda -ftype f16 -store /generations -addr :4001 >/dev/null

cleanup() {
  docker stop "$TP_APP_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "HTTPS gallery listening on all interfaces at port $TP_PORT"
echo "Local CA root: docker cp $TP_CADDY_CONTAINER:/data/talking-portrait/root.crt ./talking-portrait-root.crt"
docker run --rm --name "$TP_CADDY_CONTAINER" --network "$TP_NETWORK" \
  -p "$TP_PORT:4001" \
  -v "$TP_ROOT/server/Caddyfile:/etc/caddy/Caddyfile:ro" \
  -v talking_portrait_caddy_data:/data \
  -v talking_portrait_caddy_config:/config \
  caddy:2-alpine
