#!/usr/bin/env bash
set -euo pipefail

# The host is shared. Keep BLAS/OpenMP libraries from multiplying the four
# outer calibration workers into dozens of hidden threads.
export AUTOCALIB_OPTIMIZER_CPUS="${AUTOCALIB_OPTIMIZER_CPUS:-4}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python - <<'AUTH'
import os, hashlib, base64
from pathlib import Path
password = os.environ.get('ROC_CALIB_PASSWORD', '')
config = Path('/tmp/roc-calib-auth.conf')
if password:
    digest = base64.b64encode(hashlib.sha1(password.encode()).digest()).decode()
    Path('/tmp/roc-calib-users').write_text('roc:{SHA}' + digest + '\n')
    config.write_text('auth_basic "ROC-Calib"; auth_basic_user_file /tmp/roc-calib-users;\n')
else:
    config.write_text('')
AUTH
uvicorn roc_calib.main:app --host 0.0.0.0 --port 8000 --workers 1 &
sam_pid=$!

shutdown() {
  kill "${sam_pid}" "${web_pid:-}" "${proxy_pid:-}" 2>/dev/null || true
  wait "${sam_pid}" "${web_pid:-}" "${proxy_pid:-}" 2>/dev/null || true
}
trap shutdown EXIT INT TERM

wait_for_port() {
  local port="$1"
  local pid="$2"
  local label="$3"
  for ((attempt = 0; attempt < 300; attempt += 1)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${label} exited before becoming ready" >&2
      return 1
    fi
    if (true >"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  echo "Timed out waiting for ${label} on port ${port}" >&2
  return 1
}

cd /web
./node_modules/.bin/vinext start --hostname 127.0.0.1 --port 3001 &
web_pid=$!

# Do not expose the proxy until both upstreams accept connections. Otherwise
# requests made during the first seconds of a deployment become HTML 502s.
wait_for_port 8000 "${sam_pid}" "calibration service"
wait_for_port 3001 "${web_pid}" "web service"

nginx -g 'daemon off;' &
proxy_pid=$!

wait -n "${sam_pid}" "${web_pid}" "${proxy_pid}"
