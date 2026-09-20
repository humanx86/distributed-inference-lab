FROM nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:a3f014424ccc86cf66c84d700d2896491424205dd9457e3116ed4f0e96381311

ARG DEBIAN_FRONTEND=noninteractive

# dev utilities only
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        openssh-client \
        sudo \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        gdb \
        jq \
        ripgrep \
        less \
    && rm -rf /var/lib/apt/lists/*

# isolate Python from the system installation
ENV VIRTUAL_ENV=/home/${USERNAME}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python3.12 -m venv "${VIRTUAL_ENV}"

ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspaces
