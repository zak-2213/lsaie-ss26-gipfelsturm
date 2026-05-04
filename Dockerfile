FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.01-py3-alps3

# RUN git clone git@github.com:Dao-AILab/flash-attention.git && \
    # cd flash-attention/hopper && \
    # pip install -e . && \
    
# COPY ./flash-attention /flash-attention

# ENV FLASH_ATTENTION_FORCE_BUILD="TRUE"
# ENV FLASH_ATTENTION_DISABLE_SM80="TRUE"
# # limit parallel jobs; GH200 nodes can OOM at high parallelism
# ENV MAX_JOBS=4

# RUN cd /flash-attention/hopper && \
#     python setup.py install && \
#     cd && rm -rf /flash-attention

RUN pip install flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch2100
