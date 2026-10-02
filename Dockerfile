FROM node:22.23.2-bookworm-slim@sha256:48e4b67d85f87bd551df43704e24d252f56cc5f8e9718841aace50f19948f0f9 AS frontend
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run check && npm run build

FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 SAM2_BUILD_CUDA=0 \
    AUTOCALIB_DATA_ROOT=/data AUTOCALIB_PUBLIC_ROOT=/web/public \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends nginx ffmpeg git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=frontend /usr/local /usr/local
COPY --from=frontend /web /web
WORKDIR /service
COPY requirements.lock ./
RUN pip install -r requirements.lock
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url ${TORCH_INDEX_URL}
RUN pip install hydra-core==1.3.2 iopath==0.1.10 tqdm==4.67.1 \
    && pip install --no-deps --no-build-isolation \
       'https://github.com/facebookresearch/sam2/archive/2b90b9f5ceec907a1c18123530e92e794ad901a4.tar.gz'
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY deploy/start-autocalib.sh /usr/local/bin/start-roc-calib
COPY demo/carla /demo/carla
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["bash", "/usr/local/bin/start-roc-calib"]
