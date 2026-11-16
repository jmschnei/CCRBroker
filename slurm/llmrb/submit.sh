#!/bin/bash
# Envía los pasos de la reproducción de LLMRouterBench a Slurm (desde el nodo de login).
#
#   ./slurm/llmrb/submit.sh setup                # CPU, ~30-60 min (descargas + venv + modelo)
#   ./slurm/llmrb/submit.sh precompute           # 1 GPU, embeddings de 21 604 prompts
#   ./slurm/llmrb/submit.sh routers 42           # 1 GPU, los 5 routers con seed 42
#   ./slurm/llmrb/submit.sh routers all          # array de 5 trabajos (una semilla cada uno)
#   ./slurm/llmrb/submit.sh pipeline             # setup → precompute → routers 42 (con dependencias)
#
# Variables útiles: ROUTERS=avengers,embedllm  PARTITION_GPU=H100  EMB_MAX_SEQ=32768
#                   DEPENDENCY=afterok:<jobid>  (para encadenar a mano)
set -euo pipefail
export REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "$REPO_DIR/slurm/llmrb/config.sh"
cd "$REPO_DIR"
mkdir -p outputs/slurm
export ROUTERS EMB_MAX_SEQ EMB_BATCH TORCH_INDEX_URL TORCH_VERSION IMAGE

sb() {  # sb <nombre> <partición> <gpus> <cpus> <mem> <tiempo> [extra sbatch...] -- <paso> [args]
    local name=$1 part=$2 gpus=$3 cpus=$4 mem=$5 time=$6
    shift 6
    local extra=()
    while [ "$1" != "--" ]; do extra+=("$1"); shift; done
    shift
    [ -n "${DEPENDENCY:-}" ] && extra+=("--dependency=$DEPENDENCY")
    local gres=()
    [ "$gpus" -gt 0 ] && gres=(--gpus="$gpus")
    sbatch --parsable --export=ALL --job-name="$name" --partition="$part" ${gres[@]+"${gres[@]}"} \
        --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem" --time="$time" \
        ${extra[@]+"${extra[@]}"} slurm/llmrb/job.sbatch "$@"
}

case "${1:-}" in
setup)
    sb llmrb-setup "$PARTITION_CPU" 0 8 32G 03:00:00 -- setup
    ;;
precompute)
    sb llmrb-emb "$PARTITION_GPU" 1 8 64G 06:00:00 -- precompute
    ;;
routers)
    seed="${2:-42}"
    if [ "$seed" = "all" ]; then
        sb llmrb-routers "$PARTITION_GPU" 1 16 128G 2-00:00:00 --array=0-$((${#SEEDS[@]} - 1)) -- routers
    else
        sb llmrb-routers "$PARTITION_GPU" 1 16 128G 2-00:00:00 --export=ALL,SEED="$seed" -- routers
    fi
    ;;
pipeline)
    j1=$(sb llmrb-setup "$PARTITION_CPU" 0 8 32G 03:00:00 -- setup)
    j2=$(DEPENDENCY="afterok:$j1" sb llmrb-emb "$PARTITION_GPU" 1 8 64G 06:00:00 -- precompute)
    j3=$(DEPENDENCY="afterok:$j2" sb llmrb-routers "$PARTITION_GPU" 1 16 128G 2-00:00:00 \
        --export=ALL,SEED=42 -- routers)
    echo "setup=$j1  precompute=$j2  routers(seed 42)=$j3"
    ;;
*)
    sed -n '2,13p' "$0"
    exit 2
    ;;
esac
