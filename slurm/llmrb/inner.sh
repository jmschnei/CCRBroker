#!/bin/bash
# Pasos que se ejecutan DENTRO del contenedor (los lanza job.sbatch vía srun + pyxis).
#   inner.sh setup       CPU: uv, checkout upstream, datos, .venv, modelo de embeddings, textos
#   inner.sh precompute  GPU: rellena la caché SQLite de embeddings gte-Qwen2-7B
#   inner.sh routers     GPU: routers para una semilla (SEED o índice del array de Slurm)
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO_DIR"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1

echo "[$(date -Is)] $(hostname) paso=$1 job=${SLURM_JOB_ID:-?} array=${SLURM_ARRAY_TASK_ID:-}"

case "$1" in
setup)
    # 1) uv (gestor de entornos) en netscratch
    if ! command -v uv >/dev/null; then
        mkdir -p "$TOOLS_BIN"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$TOOLS_BIN" sh
    fi
    uv --version

    # 2) checkout de LLMRouterBench en el commit fijado + enlace results/bench
    python3 scripts/run_llmrouterbench_baselines.py setup

    # 3) datos de LLMRouterBench (1,2 GB comprimidos, 6,5 GB extraídos)
    RAW=data/raw/llmrouterbench
    if [ ! -d "$RAW/extracted/bench-release" ]; then
        mkdir -p "$RAW/extracted"
        if [ ! -f "$RAW/bench-release.tar.gz" ]; then
            curl -L --fail -C - -o "$RAW/bench-release.tar.gz.part" "$BENCH_URL"
            mv "$RAW/bench-release.tar.gz.part" "$RAW/bench-release.tar.gz"
        fi
        tar -xzf "$RAW/bench-release.tar.gz" -C "$RAW/extracted"
    fi
    ls "$RAW/extracted/bench-release" | head

    # 4) entorno de upstream con las versiones exactas del Mac
    VENV=external/LLMRouterBench/.venv
    [ -x "$VENV/bin/python" ] || uv venv --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" -r slurm/llmrb/requirements.lock.txt
    if [ -n "$TORCH_INDEX_URL" ]; then
        uv pip install --python "$VENV/bin/python" --reinstall "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
    fi
    "$VENV/bin/python" -m pip freeze 2>/dev/null >outputs/slurm/llmrb_env_freeze_cluster.txt \
        || uv pip freeze --python "$VENV/bin/python" >outputs/slurm/llmrb_env_freeze_cluster.txt

    # 5) modelo de embeddings fijado por revisión (~15 GB en $HF_HOME)
    "$LLMRB_PY" - <<'EOF'
import sys
sys.path.insert(0, "scripts")
import embedding_server as E
from huggingface_hub import snapshot_download
print(snapshot_download(E.MODEL, revision=E.REVISION))
EOF

    # 6) lista de prompts únicos a embeber (21 604)
    "$LLMRB_PY" scripts/run_llmrouterbench_baselines.py texts
    wc -l outputs/embeddings/texts.jsonl
    ;;

precompute)
    nvidia-smi
    "$LLMRB_PY" -c "import torch; assert torch.cuda.is_available(), 'sin CUDA'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
    export EMB_CACHE="$EMB_CACHE_SRC"
    "$LLMRB_PY" scripts/embedding_server.py precompute \
        --texts outputs/embeddings/texts.jsonl --batch-size "$EMB_BATCH" --max-seq-length "$EMB_MAX_SEQ"
    "$LLMRB_PY" - <<EOF
import sqlite3; print("embeddings en caché:", sqlite3.connect("$EMB_CACHE").execute("select count(*) from emb").fetchone()[0])
EOF
    ;;

routers)
    SEED="${SEED:-${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}}"
    # un puerto por trabajo: varios trabajos del array pueden caer en el mismo nodo
    export LLMRB_EMB_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
    # copia local de la caché: evita bloqueos de SQLite entre trabajos sobre el sistema de ficheros de red
    LOCAL="${TMPDIR:-/tmp}/emb_${SLURM_JOB_ID:-$$}.sqlite"
    cp "$EMB_CACHE_SRC" "$LOCAL"
    export EMB_CACHE="$LOCAL"
    trap 'rm -f "$LOCAL"' EXIT
    "$LLMRB_PY" scripts/run_llmrouterbench_baselines.py run --routers "$ROUTERS" --seeds "$SEED"
    ;;

*)
    echo "paso desconocido: $1" >&2
    exit 2
    ;;
esac
echo "[$(date -Is)] fin paso=$1"
