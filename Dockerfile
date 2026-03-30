FROM continuumio/miniconda3:23.11.0-0

# Install pytorch-bsf with MKL-backed PyTorch from the pytorch conda channel.
WORKDIR /opt/pytorch-bsf
COPY environment.yml pyproject.toml ./
COPY torch_bsf ./torch_bsf
RUN grep -v '^\s*- -e \.' environment.yml > /tmp/env.yml && \
    conda env update -n base -f /tmp/env.yml && \
    pip install --no-cache-dir --no-deps . && \
    conda clean -afy

WORKDIR /workspace
