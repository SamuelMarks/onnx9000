# ==============================================================================
# ONNX9000 Legacy & Missing Frameworks Dockerfile
# ==============================================================================
#
# This Dockerfile provides isolated environments for older frameworks (CNTK, 
# MXNet, Caffe) as well as frameworks marked as "Not Installed" in the 
# SUPPORTED_PER_FRAMEWORK.md (Keras, Paddle, XGBoost, LightGBM, H2O, LibSVM).
#
# BUILD INSTRUCTIONS:
# -------------------
# docker build -t onnx9000-legacy -f old_frameworks.debian.Dockerfile .
#
# RUN INSTRUCTIONS:
# -----------------
# To run the container interactively:
# docker run --rm -it onnx9000-legacy
#
# EXTRACTING SNAPSHOTS:
# ---------------------
# To extract the generated .json snapshot files out to the host's snapshots 
# directory, you should run the container with a volume mount that maps your local 
# snapshots folder into the container's workspace.
#
# Example command (assuming your scripts generate files in /workspace/snapshots):
#
# docker run --rm -v "$(pwd)/snapshots:/workspace/snapshots" onnx9000-legacy \
#   /bin/bash -c "source /venvs/cntk/bin/activate && python /workspace/scripts/generate_snapshots.py && \
#                 source /venvs/misc/bin/activate && python /workspace/scripts/generate_snapshots.py"
#
# Any .json file written to /workspace/snapshots inside the container will 
# automatically be saved to your host machine's snapshots folder.
#
# USAGE INSIDE THE CONTAINER:
# ---------------------------
# We use multiple virtual environments to prevent dependency conflicts.
# You can activate them using the pre-configured aliases:
#   $ activate_cntk
#   $ activate_mxnet
#   $ activate_caffe
#   $ activate_misc   <-- (for keras, paddle, xgboost, lightgbm, h2o, libsvm)
#
# ==============================================================================

FROM debian:bullseye-slim

ENV PYENV_ROOT="/opt/pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies for Pyenv, Caffe, build tools, and other frameworks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Pyenv build dependencies
        make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
        libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
        git ca-certificates \
        # Caffe via apt (easiest way on Debian without compiling from source)
        caffe python3-caffe python3-venv python3-pip \
        # Framework dependencies (OpenMPI for CNTK, libgomp1 for LightGBM/XGBoost, JRE for H2O)
        openmpi-bin libopenmpi-dev libgomp1 libquadmath0 default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/lib/x86_64-linux-gnu/libmpi_cxx.so /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libmpi.so /usr/lib/x86_64-linux-gnu/libmpi.so.12

# 2. Install Pyenv
RUN curl https://pyenv.run | bash

# 3. Install Python 3.6.15 via Pyenv
# CNTK 2.7 requires Python <= 3.6. We build it from source using pyenv.
RUN pyenv install 3.6.15 && pyenv global 3.6.15

# 4. Create multiple isolated virtual environments
RUN mkdir -p /venvs && \
    \
    # --- CNTK Environment (Python 3.6) ---
    pyenv exec python3.6 -m venv /venvs/cntk && \
    /venvs/cntk/bin/pip install --no-cache-dir --timeout 100 --retries 2 --upgrade "pip<22.0" setuptools wheel && \
    /venvs/cntk/bin/pip install --no-cache-dir --timeout 100 --retries 2 cntk==2.7 numpy==1.19.5 scipy==1.5.4 && \
    \
    # --- MXNet Environment (Python 3.6) ---
    pyenv exec python3.6 -m venv /venvs/mxnet && \
    /venvs/mxnet/bin/pip install --no-cache-dir --timeout 100 --retries 2 --upgrade "pip<22.0" setuptools wheel && \
    /venvs/mxnet/bin/pip install --no-cache-dir --timeout 100 --retries 2 mxnet numpy scipy && \
    \
    # --- Caffe Environment (System Python 3.9) ---
    # --system-site-packages is required to access the apt-installed `caffe` module
    /usr/bin/python3 -m venv --system-site-packages /venvs/caffe && \
    \
    # --- Misc Environment (System Python 3.9) ---
    # Covers remaining 'Not Installed' frameworks: keras, paddle, xgboost, lightgbm, h2o, libsvm
    /usr/bin/python3 -m venv /venvs/misc && \
    /venvs/misc/bin/pip install --no-cache-dir --timeout 100 --retries 2 --upgrade pip setuptools wheel && \
    /venvs/misc/bin/pip install --no-cache-dir --timeout 100 --retries 2 keras paddlepaddle xgboost lightgbm h2o libsvm tensorflow

# Provide helper aliases to easily activate them
RUN echo "alias activate_cntk='source /venvs/cntk/bin/activate'" >> ~/.bashrc && \
    echo "alias activate_mxnet='source /venvs/mxnet/bin/activate'" >> ~/.bashrc && \
    echo "alias activate_caffe='source /venvs/caffe/bin/activate'" >> ~/.bashrc && \
    echo "alias activate_misc='source /venvs/misc/bin/activate'" >> ~/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]