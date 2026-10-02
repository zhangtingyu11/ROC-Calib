#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host="${ROC_CALIB_HOST:-127.0.0.1}"
port="${ROC_CALIB_PORT:-3000}"
data="${ROC_CALIB_DATA:-${root}/data}"
models="${ROC_CALIB_MODELS:-${root}/.models}"
device=cuda:0
torch_index=https://download.pytorch.org/whl/cu128
gpu=(--gpus "device=${ROC_CALIB_GPU:-0}")
if [[ "${1:-}" == "--cpu" ]]; then device=cpu; gpu=(); torch_index=https://download.pytorch.org/whl/cpu;
elif [[ $# -gt 0 ]]; then echo 'Usage: ./start.sh [--cpu]' >&2; exit 2; fi
for command in docker curl sha256sum; do command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }; done
docker info >/dev/null
if [[ "$host" != 127.0.0.1 && "$host" != localhost && -z "${ROC_CALIB_PASSWORD:-}" ]]; then
  echo 'Set ROC_CALIB_PASSWORD before binding to a network interface.' >&2
  exit 1
fi
mkdir -p "$data" "$models"
data="$(cd "$data" && pwd)"
models="$(cd "$models" && pwd)"
checkpoint="$models/sam2.1_hiera_base_plus.pt"
checksum=a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5
if [[ ! -f "$checkpoint" ]]; then
  curl -fL --retry 3 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt -o "$checkpoint.partial"
  printf '%s  %s\n' "$checksum" "$checkpoint.partial" | sha256sum -c -
  mv "$checkpoint.partial" "$checkpoint"
fi
printf '%s  %s\n' "$checksum" "$checkpoint" | sha256sum -c -
docker build --network="${ROC_CALIB_BUILD_NETWORK:-default}" --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg "TORCH_INDEX_URL=$torch_index" -t roc-calib:local "$root"
if docker container inspect roc-calib >/dev/null 2>&1; then
  owner="$(docker inspect --format '{{index .Config.Labels "org.roc-calib.app"}}' roc-calib)"
  [[ "$owner" == workbench ]] || { echo 'Container name roc-calib is already in use.' >&2; exit 1; }
  docker rm -f roc-calib >/dev/null
fi
docker run -d --name roc-calib --label org.roc-calib.app=workbench \
  --restart unless-stopped --cpus="${ROC_CALIB_CPUS:-4}" "${gpu[@]}" \
  -p "$host:$port:3000" -v "$data:/data" -v "$models:/models:ro" \
  -e SAM2_DEVICE="$device" -e ROC_CALIB_PASSWORD -e ROC_CALIB_READ_ONLY \
  -e ROC_CALIB_UPLOAD_GB -e AUTOCALIB_ENABLE_LIDAR_CALIBRATION=0 \
  roc-calib:local
echo "ROC-Calib: http://$host:$port"
