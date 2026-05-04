#!/bin/bash

cd /iopsstor/scratch/cscs/$USER/lsaie-ss26-gipfelsturm/
podman build -t flashattn3:latest .
podman images
enroot import -o ../flashattn3_torch_2100.sqsh podman://flashattn3:latest
