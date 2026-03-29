FROM continuumio/miniconda3

# Install pytorch-bsf with MKL-backed PyTorch from the pytorch conda channel.
# MLflow mounts the project source into the container at runtime, so torch_bsf
# is resolved from the mount first; this layer primarily caches runtime dependencies.
WORKDIR /opt/pytorch-bsf
COPY . .
RUN grep -v '^\s*- -e \.' environment.yml > /tmp/env.yml && \
    conda env update -n base -f /tmp/env.yml && \
    pip install --no-cache-dir . && \
    conda clean -afy
