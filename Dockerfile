FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.01-py3-alps3
# FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:25.12-py3-alps3
# FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-nemo:25.11.01-alps3

# # Clone Git repo
# RUN git clone git@github.com:Dao-AILab/flash-attention.git && \
    # cd flash-attention/hopper && \
    # pip install -e . && \

# Copy git repo
COPY ./flash-attention /flash-attention

# Set some env variables for build
ENV FLASH_ATTENTION_FORCE_BUILD="TRUE"
ENV FLASH_ATTENTION_DISABLE_SM80="TRUE"
# limit parallel jobs; GH200 nodes can OOM at high parallelism
ENV MAX_JOBS=32

# # Install CUDA 13.0 toolkit and libraries
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
#     && dpkg -i cuda-keyring_1.1-1_all.deb \
#     && apt-get update && apt-get install -y --no-install-recommends \
#         cuda-toolkit-13-0 \
#         libcusparselt0 \
#         libcusparselt-dev \
#     && rm -rf /var/lib/apt/lists/* \
#     && ln -sfn /usr/local/cuda-13.0 /usr/local/cuda

# RUN pip uninstall torch torchvision torchaudio -y \
#     && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
    
# Check installed versions
# RUN nvcc --version && python -c "import torch; print(torch.__version__)"
# RUN pip list | grep torch

# Build flashattn3
RUN cd /flash-attention/hopper && python setup.py install

# # Install flash attention 3 from pre-built wheels for CUDA 13.0 and PyTorch 2.9.0 (does not have ARM support)
# RUN pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch290
# # Install flash attention 3 from pre-built wheels for CUDA 13.0 and PyTorch 2.10.0 (does not have an import flash_attn_3.flash_attn_interface)
# RUN pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch2100
