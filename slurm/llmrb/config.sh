# Configuración del cluster para la reproducción de LLMRouterBench. Se carga desde submit.sh
# (nodo de login) y desde inner.sh (dentro del contenedor). Cualquier variable se puede
# sobrescribir desde el entorno:  PARTITION_GPU=H100 ./slurm/llmrb/submit.sh precompute

# --- Rutas ------------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-/netscratch/$USER/own_broker}"
SCRATCH="${SCRATCH:-/netscratch/$USER}"

# --- Slurm / contenedor (ajustar a lo que muestren `sinfo -s` y `ls /enroot`) ------------
IMAGE="${IMAGE:-/enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh}"
PARTITION_GPU="${PARTITION_GPU:-A100-80GB}"
PARTITION_CPU="${PARTITION_CPU:-batch}"
CONTAINER_MOUNTS="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$REPO_DIR:$REPO_DIR"

# --- Experimento ------------------------------------------------------------------------
# Routers y semillas (protocolo del apéndice B del paper).
ROUTERS="${ROUTERS:-avengers,embedllm,graphrouter,avengers_pro,routellm}"
SEEDS=(42 999 2024 2025 3407)
# Truncado de los embeddings en tokens del modelo. 8192 = igual que en el Mac (permite mezclar
# cachés y resultados); 32768 = sin truncado propio (upstream trunca a 25 000 tokens tiktoken).
export EMB_MAX_SEQ="${EMB_MAX_SEQ:-8192}"
EMB_BATCH="${EMB_BATCH:-16}"
if [ "$EMB_MAX_SEQ" = "8192" ]; then
    EMB_CACHE_SRC="$REPO_DIR/outputs/embeddings/gte_qwen2_7b.sqlite"
else
    EMB_CACHE_SRC="$REPO_DIR/outputs/embeddings/gte_qwen2_7b_L${EMB_MAX_SEQ}.sqlite"
fi
# Opcional: índice de PyTorch si el driver del cluster no soporta la build CUDA por defecto de
# torch==2.15.0 (p. ej. https://download.pytorch.org/whl/cu126) y la versión a instalar desde él.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TORCH_VERSION="${TORCH_VERSION:-2.15.0}"

# --- Cachés fuera de $HOME (todo en netscratch) -----------------------------------------
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRATCH/.local/share/uv/python}"
TOOLS_BIN="$SCRATCH/.local/bin"
export PATH="$TOOLS_BIN:$PATH"

LLMRB_PY="$REPO_DIR/external/LLMRouterBench/.venv/bin/python"
BENCH_URL="https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz"
