#!/usr/bin/env bash
# ============================================================================
# gpu_setup.sh — GPU discovery + vLLM setup + model pinning
#
# Uso:
#   ./gpu_setup.sh --setup          (solo instala dependencias)
#   ./gpu_setup.sh --discover        (descubre GPUs)
#   ./gpu_setup.sh --pin <model>     (pinna una revision de HF)
#   ./gpu_setup.sh --bench <model>   (mide tok/s y precio)
#   ./gpu_setup.sh --all             (hace todo)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # CCRBroker repo root
VENV="$REPO_ROOT/.venv/bin/activate"

LOG() { echo "[gpu_setup] $(date +%F_%T) $*"; }
ERROR() { echo "[gpu_setup] ERROR: $*" >&2; exit 1; }

# ---------- GPU discovery ---------------------------------------------------
discover_gpus() {
    LOG "=== Descubriendo GPUs ==="

    if ! command -v nvidia-smi &>/dev/null; then
        ERROR "nvidia-smi no encontrado. ¿Tenés CUDA installed?"
    fi

    # Summary
    echo ""
    echo "GPU Count: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"
    echo ""

    nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,cuda_version \
        --format=csv,noheader,nounits

    echo ""
    echo "PCIe topology:"
    nvidia-smi topo -m 2>/dev/null || echo "  (no disponible)"
    echo ""

    echo "CUDA drivers:"
    nvcc --version 2>/dev/null || echo "  nvcc no encontrado"
    echo ""
}

# ---------- vLLM installation -----------------------------------------------
install_vllm() {
    LOG "=== Instalando vLLM y deps de evaluación ==="

    if [ ! -f "$VENV" ]; then
        ERROR "venv no existe en $REPO_ROOT. Correr 'make install' primero."
    fi

    source "$VENV"

    # Install base + vllm extras
    pip install -q "vllm>=0.6.0" 2>&1 | tail -5 || true

    # Also install evaluation deps
    pip install -q "datasets>=2.16" "rouge-score" "sacrebleu" "jiwer" "spacy" 2>&1 | tail -5 || true

    # Verify installation
    python -c "import vllm; print('vLLM version:', vllm.__version__)" 2>/dev/null || {
        ERROR "vLLM installation failed. Check CUDA compatibility."
    }

    LOG "vLLM installed: $(python -c 'import vllm; print(vllm.__version__)')"
}

# ---------- Model pinning ---------------------------------------------------
pin_model() {
    local model="$1"
    LOG "=== Pinning revision de HF para: $model ==="

    if [ -z "$model" ]; then
        ERROR "Uso: gpu_setup.sh --pin <model_name>"
    fi

    source "$VENV"

    # Find the HF hub entry for this model
    python3 << PYEOF
import requests, json

model = "$model"
url = f"https://huggingface.co/api/models/{model}"
try:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    sha = data.get("sha", "UNKNOWN")
    last_modified = data.get("lastModified", "UNKNOWN")

    print(f"Modelo: {model}")
    print(f"Revision (commit SHA): {sha[:12]}...")
    print(f"Última actualización: {last_modified}")
    print(f"Tags: {', '.join(data.get('tags', [])[:5])}")
    print()

    # Write to pin file
    import os, sys
    repo_dir = os.path.dirname(sys.argv[0]) if len(sys.argv) > 1 else "."
    pin_file = os.path.expanduser("~/.codex/gpu_model_pins.json")

    pins = {}
    if os.path.exists(pin_file):
        with open(pin_file) as f:
            pins = json.load(f)

    pins[model] = {"sha": sha, "pinned_at": last_modified}
    with open(pin_file, "w") as f:
        json.dump(pins, f, indent=2)

    print(f"Revision pinned to: {pin_file}")
except Exception as e:
    print(f"Error fetching model info: {e}", file=sys.stderr)
    exit(1)
PYEOF
}

# ---------- Benchmark --------------------------------------------------------
benchmark_model() {
    local model="$1"

    if [ -z "$model" ]; then
        ERROR "Uso: gpu_setup.sh --bench <model_name>"
    fi

    source "$VENV"

    python3 << PYEOF
import time, json, sys
from pathlib import Path

model = "$model"
print(f"\nBenchmarking {model}...")
print("=" * 60)

# Load vLLM
from vllm import LLM, SamplingParams
import numpy as np

# Warmup (2 small batches to init CUDA)
print("  [warmup] Inicializando modelo en GPU...")
llm = LLM(model=model, tensor_parallel_size=1, dtype="bfloat16", max_model_len=4096)
tokenizer = llm.get_tokenizer()

# Test with a few prompts
test_prompts = ["The quick brown fox jumps over the lazy dog."] * 4
print(f"  [warmup] Completando {len(test_prompts)} prompts...")
sps = SamplingParams(temperature=0.2, max_tokens=256, seed=42)
llm.generate(test_prompts, sps)
del test_prompts  # free

# Benchmark configs: batch sizes and output lengths
benchmarks = [
    (1, 64),   # 1 req, 64 tokens out
    (4, 64),   # 4 req, 64 tokens out
    (8, 256),  # 8 req, 256 tokens out
    (16, 512), # 16 req, 512 tokens out
    (32, 1024),# 32 req, 1024 tokens out
]

results = []
for batch_size, max_out in benchmarks:
    prompts = ["What is the capital of France? Answer in one sentence."] * batch_size

    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.2, max_tokens=max_out, seed=42))
    t1 = time.time()

    dt = t1 - t0
    total_in = sum(len(o.prompt_token_ids) for o in outs)
    total_out = sum(sum(c.token_ids for c in o.outputs) for o in outs)

    # Token throughput metrics
    input_tps = total_in / (dt * 1e6) if dt > 0 else 0  # M tokens/sec
    output_tps = total_out / dt if dt > 0 else 0        # tokens/sec

    latency_per_req_ms = (dt / batch_size) * 1000 if batch_size > 0 else 0

    results.append({
        "batch": batch_size,
        "max_out_tokens": max_out,
        "latency_s": round(dt, 3),
        "latency_per_req_ms": round(latency_per_req_ms, 1),
        "output_tps": round(output_tps, 1),
        "input_tok_mb_sec": round(input_tps, 2),
    })

print(f"\n{'Batch':>6} | {'MaxOut':>8} | {'Latency(s)':>10} | {'Req/ms':>10} | {'Output TPS':>10}")
print("-" * 60)
for r in results:
    print(f"{r['batch']:>6} | {r['max_out_tokens']:>8} | {r['latency_s']:>10.3f} | "
          f"{1000/r['latency_per_req_ms']:>10.1f} | {r['output_tps']:>10.1f}")

# GPU memory usage
import subprocess
mem_info = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True)
if mem_info.returncode == 0:
    used, total = [int(x.strip()) for x in mem_info.stdout.strip().split('\n')[0].split(',')]
    print(f"\nGPU Memory: {used}/{total} MiB")

# Estimate cost per request (assuming ~$0.03/hr per A100)
gpu_cost_per_hour = 0.50  # USD/hour for rented A100
avg_latency_s = np.mean([r['latency_s'] / r['batch'] for r in results[2:]])
cost_per_request = (avg_latency_s / 3600) * gpu_cost_per_hour

print(f"\nEstimado de coste (A100 @ ${gpu_cost_per_hour}/hr):")
print(f"  Latencia promedio/req: {avg_latency_s:.3f}s")
print(f"  Coste por request: \${cost_per_request*1000:.4f}m ({cost_per_request:.6f} USD)")

# Save results
pin_file = Path.home() / ".codex" / "gpu_model_pins.json"
pins = {}
if pin_file.exists():
    with open(pin_file) as f:
        pins = json.load(f)

benchmark_key = model
benchmarks_data = pins.get("_benchmarks", {})
benchmarks_data[benchmark_key] = {
    "latency_avg_s": round(avg_latency_s, 3),
    "results": results,
    "date": time.strftime("%Y-%m-%d")
}
pins["_benchmarks"] = benchmarks_data

with open(pin_file, "w") as f:
    json.dump(pins, f, indent=2)

print(f"\nResultados guardados en {pin_file}")
PYEOF
}

# ---------- GPU info display --------------------------------------------------
show_gpu_info() {
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,cuda_version \
        --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r idx name mem_total mem_used mem_free drv cuda; do
        printf "GPU %s: %-40s MEM %3d%% used (%s/%d MiB) CUDA %s\n" \
            "$idx" "$name" "$((mem_used * 100 / mem_total))" "$mem_used" "$mem_total" "$cuda"
    done
}

# ---------- Main -----------------------------------------------------------------
case "${1:-}" in
    --setup)
        install_vllm
        ;;
    --discover)
        discover_gpus
        ;;
    --pin)
        pin_model "${2:?Uso: $0 --pin <model_name>}"
        ;;
    --bench)
        benchmark_model "${2:?Uso: $0 --bench <model_name>}"
        ;;
    --all)
        discover_gpus
        install_vllm
        echo ""
        echo "Setup completado. Ahora podés:"
        echo "  ./gpu_setup.sh --pin <model_hf_name>  # pinna revision"
        echo "  ./gpu_setup.sh --bench <model_hf_name>  # mide throughput"
        ;;
    --info)
        show_gpu_info
        ;;
    *)
        cat <<USAGE
Uso: $0 {--setup|--discover|--pin <model>|--bench <model>|--all|--info}

Opciones:
  --setup       Instala vLLM + deps de evaluación en el venv
  --discover    Muestra GPUs disponibles y su info básica
  --pin <m>     Pinea la revision de HF para un modelo (para reproducibilidad)
  --bench <m>   Mide throughput (tok/s) de un modelo con vLLM
  --all         Ejecuta discover + setup
  --info        Muestra información rápida de GPUs

Ejemplos:
  $0 --all                          # discovery + instalación completa
  $0 --pin Qwen/Qwen3-30B-A3B-Instruct-2507
  $0 --bench Qwen/Qwen3-30B-A3B-Instruct-2507

USAGE
        exit 1
        ;;
esac
