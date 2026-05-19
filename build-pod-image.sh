#!/bin/bash
#SBATCH --job-name=LSAIE_build-pod-image
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:29:00
#SBATCH --output=logs/build_pod_image_%j.out
#SBATCH --error=logs/build_pod_image_%j.out

set -e

log() {
    echo "[$(date)] $1"    
}
success() {
    log "SUCCESS: $1"
}
error () {
    log "ERROR: $1"
}

# Build the image using the Dockerfile in the current directory
cd /iopsstor/scratch/cscs/$USER/lsaie-ss26-gipfelsturm/
podman build -t flashattn3:latest .
success "Image built."

# Verify that this line works in this image:
# from flash_attn_3.flash_attn_interface import _flash_attn_forward
podman run --rm flashattn3:latest python -c "from flash_attn_3.flash_attn_interface import _flash_attn_forward; print('FlashAttention 3 imported successfully!')"
success "Image verified successfully."

# Export image to .sqsh file for later training environments
podman images
DATE=$(date +%Y%m%d_%H%M%S)
enroot import -o ../flashattn3_$DATE.sqsh podman://flashattn3:latest
success "Image exported to ../flashattn3_$DATE.sqsh."

