FROM python:3.12-slim

# Install system build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install pytorch-bsf from the current source tree so dependencies match this commit.
# MLflow mounts the project source into the container at runtime, and Python resolves
# torch_bsf from that mount first, so this installation primarily serves as a
# dependency cache while remaining in sync with the repo being built.
WORKDIR /opt/pytorch-bsf
COPY . .
RUN pip install --no-cache-dir .
