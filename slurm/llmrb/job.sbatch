#!/bin/bash
# Envoltorio genérico: ejecuta un paso de inner.sh dentro del contenedor (pyxis/enroot).
# No lanzar directamente: usar slurm/llmrb/submit.sh, que fija partición, recursos y entorno.
#SBATCH --output=outputs/slurm/%x-%A_%a.out
set -euo pipefail
source "$REPO_DIR/slurm/llmrb/config.sh"

srun -K \
    --container-image="$IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir="$REPO_DIR" \
    bash "$REPO_DIR/slurm/llmrb/inner.sh" "$@"
