# Installation

## Solver

Python 3.12 on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock -e .
```

On Debian/Ubuntu, install the system's `python3-venv` package if `venv` reports that `ensurepip` is missing.

The command-line solver runs on CPU. Point selection and image segmentation are separate from fitting saved pairs.

## Workbench

Install Docker. GPU mode also needs a compatible NVIDIA driver and NVIDIA Container Toolkit.

```bash
./start.sh              # NVIDIA GPU; downloads SAM2 weights on first use
./start.sh --cpu        # CPU segmentation; slower for large images
```

The first build downloads Python, Node, PyTorch and SAM2 dependencies. Leave several GB of free disk space. Dependencies and source revisions are fixed in the Dockerfile and lock files.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ROC_CALIB_PORT` | `3000` | Browser port |
| `ROC_CALIB_DATA` | `./data` | Groups, inputs, cache and exports |
| `ROC_CALIB_MODELS` | `./.models` | SAM2 checkpoint |
| `ROC_CALIB_GPU` | `0` | NVIDIA device |
| `ROC_CALIB_CPUS` | `4` | Container CPU limit |
| `ROC_CALIB_UPLOAD_GB` | `20` | Maximum size of one upload |

Restart by running the same command. Stop with `docker stop roc-calib`. View logs with `docker logs roc-calib`.

## Remote access

For personal use, leave the server bound to localhost and forward its port:

```bash
ssh -L 3000:127.0.0.1:3000 user@server
```

For a shared installation, set `ROC_CALIB_HOST=0.0.0.0` and `ROC_CALIB_PASSWORD`. The login name is `roc`. Use an HTTPS reverse proxy. This is a shared workspace, not a multi-tenant service.

`ROC_CALIB_READ_ONLY=1` disables all API changes and calculations. Use a separate data directory for a public demonstration.

## Frontend development

Node 22.23.2 or newer:

```bash
cd web
npm ci
npm run dev
```

The dev server forwards API requests to localhost:8000. Run `uvicorn roc_calib.main:app --port 8000` with `AUTOCALIB_DATA_ROOT` set to a writable directory. Manual drawing works without SAM2; SAM-assisted drawing needs the model dependencies used by the Docker image.
