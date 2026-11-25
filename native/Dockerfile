FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libeigen3-dev libopencv-dev libssl-dev libblas-dev liblapack-dev && rm -rf /var/lib/apt/lists/*
COPY native /src
RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release && cmake --build /build --target autocalib_native -j4
ENTRYPOINT ["/build/autocalib_native"]
