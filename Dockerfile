FROM python:3.12-slim

# Install system build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Pre-install pytorch-bsf from PyPI to cache all runtime dependencies.
# When MLflow runs a project with this image it mounts the project source
# into the container and Python resolves torch_bsf from that mount first,
# so the pre-installed copy serves only as a dependency cache.
RUN pip install --no-cache-dir pytorch-bsf
