#!/usr/bin/env bash
# ============================================================================
# generate_run.sh — Generación de datasets (Fases 2-5) con vLLM en GPU/SLURM
#
# Flujo:
#   1) Descubre GPUs y verifica compatibilidad
#   2) Carga modelos pinned desde ~/.codex/gpu_model_pins.json
#   3) Ejecuta generación para cada fase (ConfigRoute → DriftRoute → AgentRoute → AdvRoute)
#   4) Guarda resultados en outputs/ + logs/
#   5) Soporte nativo para SLURM con auto-detección de jobs por GPU
#
# Uso:
#   ./generate_run.sh --setup            (instala deps y verifica entorno)
#   ./generate_run.sh --discover         (descubre GPUs y muestra info)
#   ./generate_run.sh --run <FASE>       (genera datos para una fase específica)
#   ./generate_run.sh --run-all          (genera todas las fases en secuencia)
#   ./generate_run.sh --slurm <PHASE>    (submit a SLURM cluster)
#   ./generate_run.sh --budget           (estima presupuesto con mediciones reales)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/.venv/bin/activate"
PIN_FILE="$HOME/.codex/gpu_model_pins.json"
OUTPUT_BASE="$REPO_ROOT/outputs"
LOG_DIR="$REPO_ROOT/logs"

# Colores para output (si terminal lo soporta)
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    GREEN='' RED='' YELLOW='' BOLD='' NC=''
fi

LOG() { echo -e "[${GREEN}generate_run${NC}] $(date +%F_%T) $*"; }
ERROR() { echo -e "[${RED}ERROR${NC}] $*" >&2; exit 1; }
WARN() { echo -e "[${YELLOW}WARN${NC}] $*"; }
INFO() { echo -e "[${BOLD}INFO${NC}] $*"; }

# ---------- GPU discovery ---------------------------------------------------
discover_gpus() {
    INFO "=== Descubriendo GPUs ==="

    if ! command -v nvidia-smi &>/dev/null; then
        ERROR "nvidia-smi no encontrado. ¿Tenés CUDA instalado y los drivers activos?"
    fi

    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo ""
    echo "GPU Count: $gpu_count"
    echo ""

    nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,cuda_version \
        --format=csv,noheader,nounits

    echo ""
    echo "PCIe topology:"
    nvidia-smi topo -m 2>/dev/null || echo "  (no disponible)"
    echo ""

    # Check CUDA
    if command -v nvcc &>/dev/null; then
        echo "CUDA drivers: $(nvcc --version 2>/dev/null | grep release || echo 'unknown')"
    else
        echo "CUDA drivers: nvcc no encontrado (puede funcionar igual con vLLM)"
    fi
    echo ""
}

# ---------- Model pinning ---------------------------------------------------
load_pinned_models() {
    if [[ ! -f "$PIN_FILE" ]]; then
        echo "{}"
        return
    fi
    cat "$PIN_FILE"
}

get_pinned_shas() {
    local model="$1"
    load_pinned_models | python3 -c "
import sys, json
pins = json.load(sys.stdin)
if '$model' in pins:
    print(pins['$model']['sha'][:12])
else:
    print('NOT_PINNED')
" 2>/dev/null || echo "ERROR"
}

# ---------- Setup -------------------------------------------------------------
do_setup() {
    LOG "=== Instalando dependencias de generación ==="

    if [[ ! -f "$VENV" ]]; then
        ERROR "venv no encontrado en $REPO_ROOT. Ejecutar 'make install' primero."
    fi

    source "$VENV"

    # Install base + vllm extras + evaluation deps
    LOG "Instalando vLLM y deps..."
    pip install -q "vllm>=0.6.0" "datasets>=2.16" "rouge-score" "sacrebleu" "jiwer" 2>&1 | tail -3

    # Verify
    python3 -c "import vllm; print('vLLM version:', vllm.__version__)" 2>/dev/null || {
        ERROR "vLLM installation failed. Verificar compatibilidad CUDA."
    }

    LOG "Dependencias instaladas: $(python3 -c 'import vllm; print(vllm.__version__)')"
}

# ---------- Benchmark model (get real tok/s) ---------------------------------
benchmark_model() {
    local model="$1"
    INFO "Benchmarking $model..."

    source "$VENV"
    python3 << PYEOF
import time, json, sys, os
from pathlib import Path

model = "$model"
print(f"\nBenchmarking {model}...")
print("=" * 60)

from vllm import LLM, SamplingParams
import numpy as np

# Warmup
print("  [warmup] Loading model into GPU...")
llm = LLM(model=model, tensor_parallel_size=1, dtype="bfloat16", max_model_len=4096)
tokenizer = llm.get_tokenizer()

test_prompts = ["The quick brown fox jumps over the lazy dog."] * 4
sps = SamplingParams(temperature=0.2, max_tokens=256, seed=42)
llm.generate(test_prompts, sps)
del test_prompts

# Benchmark configs: (batch_size, max_out_tokens)
benchmarks = [
    (1, 64),   (4, 64),   (8, 256),   (16, 512),   (32, 1024)
]

results = []
for batch_size, max_out in benchmarks:
    prompts = ["What is the capital of France? Answer in one sentence."] * batch_size

    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.2, max_tokens=max_out, seed=42))
    dt = time.time() - t0

    total_in = sum(len(o.prompt_token_ids) for o in outs)
    total_out = sum(sum(c.token_ids for c in o.outputs) for o in outs)

    output_tps = total_out / dt if dt > 0 else 0
    latency_per_req_ms = (dt / batch_size) * 1000 if batch_size > 0 else 0

    results.append({
        "batch": batch_size,
        "max_out_tokens": max_out,
        "latency_s": round(dt, 3),
        "latency_per_req_ms": round(latency_per_req_ms, 1),
        "output_tps": round(output_tps, 1),
    })

print(f"\n{'Batch':>6} | {'MaxOut':>8} | {'Latency(s)':>10} | {'Req/ms':>10} | {'Output TPS':>10}")
print("-" * 60)
for r in results:
    print(f"{r['batch']:>6} | {r['max_out_tokens']:>8} | {r['latency_s']:>10.3f} | "
          f"{1000/r['latency_per_req_ms']:>10.1f} | {r['output_tps']:>10.1f}")

# Estimate cost per request (assume ~$0.50/hr for rented A100)
gpu_cost_per_hour = 0.50
avg_latency_s = np.mean([r['latency_s'] / r['batch'] for r in results[2:]])
cost_per_request = (avg_latency_s / 3600) * gpu_cost_per_hour

print(f"\nEstimated cost (A100 @ \${gpu_cost_per_hour}/hr):")
print(f"  Avg latency/req: {avg_latency_s:.3f}s")
print(f"  Cost per request: \${cost_per_request*1000:.4f}m ({cost_per_request:.6f} USD)")

# Save results
pin_file = Path(os.path.expanduser("~/.codex")) / "gpu_model_pins.json"
pins = {}
if pin_file.exists():
    with open(pin_file) as f:
        pins = json.load(f)

benchmarks_data = pins.get("_benchmarks", {})
benchmarks_data[model] = {
    "latency_avg_s": round(avg_latency_s, 3),
    "results": results,
    "date": time.strftime("%Y-%m-%d")
}
pins["_benchmarks"] = benchmarks_data

with open(pin_file, "w") as f:
    json.dump(pins, f, indent=2)

print(f"\nResults saved to {pin_file}")
PYEOF
}

# ---------- Generation runner ------------------------------------------------
run_generation() {
    local phase="$1"
    shift
    local n_samples="${1:-200}"
    local warmup="${2:-5}"

    INFO "=== Running Phase $phase (N=$n_samples) ==="

    if [[ ! -f "$VENV" ]]; then
        ERROR "venv not found. Run './generate_run.sh --setup' first."
    fi

    source "$VENV"

    # Load pinned models for this phase
    local phase_models
    case "$phase" in
        2) phase_models=("Qwen/Qwen3-30B-A3B-Instruct-2507" "meta-llama/Llama-3.1-70B-Instruct") ;;
        3) phase_models=("Qwen/Qwen3-30B-A3B-Instruct-2507") ;;  # DriftRoute uses one model over time
        4) phase_models=("Qwen/Qwen3-30B-A3B-Instruct-2507" "meta-llama/Llama-3.1-8B-Instruct") ;;
        5) phase_models=("Qwen/Qwen3-30B-A3B-Instruct-2507") ;;
        *) ERROR "Invalid phase: $phase (expected 2-5)" ;;
    esac

    # Check pinned revisions
    for model in "${phase_models[@]}"; do
        local sha
        sha=$(get_pinned_shas "$model")
        if [[ "$sha" == "NOT_PINNED" || "$sha" == "ERROR" ]]; then
            WARN "Model $model is not pinned. Pinning now..."
            # Use latest commit from HF
            local latest_sha
            latest_sha=$(python3 -c "
import requests, sys
try:
    r = requests.get(f'https://huggingface.co/api/models/$model', timeout=30)
    if r.status_code == 200:
        print(r.json()['sha'][:12])
    else:
        sys.exit(1)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || echo "FAILED")

            if [[ "$latest_sha" == "FAILED" ]]; then
                ERROR "Failed to fetch revision for $model. Check network/HF availability."
            fi

            # Update pin file
            python3 -c "
import json, time
from pathlib import Path
pin_file = Path('$PIN_FILE')
pins = {}
if pin_file.exists():
    with open(pin_file) as f:
        pins = json.load(f)
pins['$model'] = {'sha': '$latest_sha', 'pinned_at': time.strftime('%Y-%m-%d')}
with open(pin_file, 'w') as f:
    json.dump(pins, f, indent=2)
print('Pinned to:', '$latest_sha')
" 2>/dev/null || true
        fi
    done

    # Run generation
    local out_dir="$OUTPUT_BASE/phase$phase"
    mkdir -p "$out_dir" "$LOG_DIR"

    INFO "Output directory: $out_dir"
    INFO "Starting generation..."

    python3 << PYEOF 2>&1 | tee "$LOG_DIR/phase${phase}_$(date +%F_%T).log"
import json, os, sys, time, pandas as pd, numpy as np
from pathlib import Path

sys.path.insert(0, "$REPO_ROOT")

from routedata.schema import load_table, validate
from generate.configroute import sample_queries, build_config_grid
from generate.backends import VLLMBackend, MockBackend

PHASE = "$phase"
N_SAMPLES = $n_samples
WARMUP = $warmup
OUT_DIR = Path("$out_dir")

# Load source queries
source_queries = load_table("$OUTPUT_BASE/phase2/query_sample").query if PHASE == "2" else None
if source_queries is None:
    # Use combined queries from dataset
    q_df, _, _ = pd.read_parquet("data/processed/unified/queries_combined.parquet"), None, None
    print(f"Using {len(q_df)} queries from unified dataset")

# Load pinned models
pin_file = Path(os.path.expanduser("~/.codex/gpu_model_pins.json"))
pins = {}
if pin_file.exists():
    with open(pin_file) as f:
        pins = json.load(f)

models_pinned = {}
for key, val in pins.items():
    if key not in ("_benchmarks",):
        models_pinned[key] = {"sha": val["sha"]}

print(f"Loaded {len(models_pinned)} pinned models from {pin_file}")

# Create config grid for Phase 2 (ConfigRoute)
if PHASE == "2":
    budgets = [None, 128, 256, 512]  # token budgets
    efforts = ["nothink", "think_low", "think_mid", "think_high"]

    config_models = {}
    for model_name, pin_info in models_pinned.items():
        config_models[model_name] = {
            "revision": pin_info["sha"],
            "price_in": 0.0,  # Will be set from benchmark data
            "price_out": 0.0,
            "thinking": True if "think_low" in efforts else False,
            "price_source": "selfhost"
        }

    configs_df = build_config_grid(config_models, budgets=budgets, efforts=efforts)
    print(f"Generated {len(configs_df)} config entries for Phase 2")

    # Sample queries (stratified)
    if source_queries is None:
        q_df = pd.read_parquet("data/processed/unified/queries_combined.parquet")
        quotas = {"routerbench": N_SAMPLES, "sprout": N_SAMPLES, "llmrouterbench": N_SAMPLES}
        sampled = sample_queries(
            sources={"combined": q_df},
            quotas={"combined": len(q_df)},
            strat=("domain", "difficulty"),
            seed=42
        )
    else:
        sampled = source_queries

    print(f"Sampled {len(sampled)} queries for generation")

    # Generate with VLLMBackend (or Mock if no GPU)
    try:
        from vllm import LLM
        has_gpu = True
    except ImportError:
        has_gpu = False

    backend_type = "vllm" if has_gpu else "mock"
    print(f"Using {backend_type} backend")

    # Create output directory structure
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run generation per model
    for model_name in models_pinned.keys():
        revision = models_pinned[model_name]["sha"]
        print(f"\nGenerating for {model_name} (revision: {revision[:12]})...")

        if has_gpu:
            backend_factory = lambda m, rev: VLLMBackend(m, rev)
        else:
            skill = 0.7  # default skill for mock
            backend_factory = lambda m, rev: MockBackend(m, skill=skill)

        # Import and run generation logic
        from generate.configroute import run

        results_df = run(
            queries=sampled,
            configs=configs_df,
            backend_factory=backend_factory,
            out_dir=str(OUT_DIR),
            run_id=f"run0",
            snapshot_id="s0",
            batch=32,  # Adjust based on VRAM
            seed=42,
            log=print
        )

        print(f"Generated {len(results_df)} responses for {model_name}")

        # Save summary
        summary = {
            "phase": PHASE,
            "model": model_name,
            "revision": revision,
            "num_configs": len(configs_df),
            "num_queries": len(sampled),
            "num_responses": len(results_df),
            "backend": backend_type,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        with open(OUT_DIR / f"summary_{model_name.replace('/', '_')}.json", "w") as f:
            json.dump(summary, f, indent=2)

print(f"\nPhase {PHASE} generation complete!")
PYEOF

    INFO "Phase $phase completed. Results in $out_dir"
}

# ---------- SLURM job submission ---------------------------------------------
submit_slurm() {
    local phase="$1"

    if ! command -v sinfo &>/dev/null; then
        ERROR "sinfo not found. Are you on a SLURM cluster?"
    fi

    INFO "Submitting Phase $phase to SLURM..."

    # Get available GPUs
    local gpu_info
    gpu_info=$(sinfo -N -h -o "%n %G" | head -10)
    echo "Available nodes:"
    echo "$gpu_info"

    # Submit job script
    local job_script="$LOG_DIR/submit_phase${phase}_$(date +%F_%T).sh"

    cat > "$job_script" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=generate_phase${phase}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=$LOG_DIR/slurm_%j.out
#SBATCH --error=$LOG_DIR/slurm_%j.err

set -euo pipefail

# Load environment
source /path/to/your/venv/bin/activate  # Adjust path as needed

cd $REPO_ROOT
./generate_run.sh run ${phase} \$1 \$2

SLURM_EOF

    chmod +x "$job_script"

    sbatch "$job_script" "${@:2}"  # Pass additional args to generation

    INFO "Job submitted. Check status with 'squeue -u $USER'"
}

# ---------- Budget estimation -------------------------------------------------
estimate_budget() {
    INFO "Estimating budget for Phase 2 (ConfigRoute)..."

    source "$VENV"
    python3 << 'PYEOF'
import json, time, os
from pathlib import Path

pin_file = Path(os.path.expanduser("~/.codex/gpu_model_pins.json"))
pins = {}
if pin_file.exists():
    with open(pin_file) as f:
        pins = json.load(f)

benchmarks = pins.get("_benchmarks", {})

print("\n" + "=" * 60)
print("BUDGET ESTIMATION FOR PHASE 2 (ConfigRoute)")
print("=" * 60)

if not benchmarks:
    print("\nNo benchmark data found. Run './generate_run.sh --bench <model>' first.")
    exit(1)

# Config for Phase 2
num_configs = 75  # from pipeline analysis
num_queries = 27_203  # from LLMRouterBench (largest source)
avg_out_tokens = 256  # average output tokens per request
batch_size = 16  # typical batch size

print(f"\nPhase 2 Configuration:")
print(f"  Queries: {num_queries:,}")
print(f"  Configs: {num_configs}")
print(f"  Total requests: {num_queries * num_configs:,}")
print(f"  Avg out tokens/req: {avg_out_tokens}")
print(f"  Batch size: {batch_size}")

# Estimate GPU hours using benchmark data
total_gpu_hours = 0
for model_name, bench_data in benchmarks.items():
    results = bench_data.get("results", [])
    if not results:
        print(f"\n{model_name}: No benchmark data")
        continue

    # Use the 32-batch config for estimation
    thirty_two_batch = next((r for r in results if r["batch"] == 32), None)
    if not thirty_two_batch:
        # Extrapolate from available data
        best = max(results, key=lambda r: r["batch"])
        latency_s = best["latency_s"] / best["batch"]
        output_tps = best["output_tps"]
    else:
        latency_s = thirty_two_batch["latency_s"] / 32
        output_tps = thirty_two_batch["output_tps"]

    # Calculate GPU hours for this model
    total_requests = num_queries * num_configs
    total_latency_s = total_requests * latency_s
    gpu_hours = total_latency_s / 3600

    print(f"\n{model_name}:")
    print(f"  Avg latency/req: {latency_s:.3f}s")
    print(f"  Output TPS: {output_tps}")
    print(f"  GPU hours (1x A100): {gpu_hours:.1f}")

    total_gpu_hours += gpu_hours

print(f"\n{'=' * 60}")
print(f"TOTAL ESTIMATED GPU HOURS (all models): {total_gpu_hours:.1f}")
print(f"ESTIMATED COST (@ \$0.50/hr per A100): \${total_gpu_hours * 0.50:.2f}")
print(f"{'=' * 60}")

# Suggest optimizations
print(f"\nOptimization suggestions:")
print(f"  - Use tensor parallelism for models >80B params")
print(f"  - Consider reducing configs from {num_configs} to ~50-60 if VRAM is tight")
print(f"  - Process phases sequentially to save GPU time between runs")
PYEOF
}

# ---------- Main -----------------------------------------------------------------
case "${1:-}" in
    --setup)
        do_setup
        ;;
    --discover)
        discover_gpus
        ;;
    --run)
        [[ -z "${2:-}" ]] && { ERROR "Uso: $0 --run <PHASE>"; exit 1; }
        run_generation "$2" "${3:-200}" "${4:-5}"
        ;;
    --run-all)
        do_setup || true
        for phase in 2 3 4 5; do
            LOG "=== Starting Phase $phase ==="
            run_generation "$phase" "${1:-200}" "${2:-5}"
            LOG "Phase $phase completed successfully"
            echo ""
        done
        ;;
    --slurm)
        [[ -z "${2:-}" ]] && { ERROR "Uso: $0 --slurm <PHASE>"; exit 1; }
        submit_slurm "$2" "${@:3}"
        ;;
    --budget)
        estimate_budget
        ;;
    --pin)
        if [[ -z "${2:-}" ]]; then
            ERROR "Uso: $0 --pin <model_name>"
        fi

        source "$VENV"
        python3 << PYEOF
import requests, json, time, os
from pathlib import Path

model = "$2"
print(f"Pinning revision for {model}...")

try:
    r = requests.get(f"https://huggingface.co/api/models/{model}", timeout=30)
    r.raise_for_status()
    data = r.json()

    sha = data.get("sha", "UNKNOWN")
    last_modified = data.get("lastModified", "UNKNOWN")

    pin_file = Path(os.path.expanduser("~/.codex")) / "gpu_model_pins.json"
    pins = {}
    if pin_file.exists():
        with open(pin_file) as f:
            pins = json.load(f)

    pins[model] = {"sha": sha, "pinned_at": last_modified}
    with open(pin_file, "w") as f:
        json.dump(pins, f, indent=2)

    print(f"Pinned to: {sha[:12]}... ({last_modified})")
except Exception as e:
    print(f"Error: {e}", file=__import__('sys').stderr)
    exit(1)
PYEOF
        ;;
    --bench)
        if [[ -z "${2:-}" ]]; then
            ERROR "Uso: $0 --bench <model_name>"
        fi
        benchmark_model "$2"
        ;;
    --info)
        discover_gpus
        ;;
    *)
        cat <<USAGE
Usage: $0 {--setup|--discover|--run <PHASE>|--run-all|--slurm <PHASE>|--budget|--pin <model>|--bench <model>|--info}

Options:
  --setup       Installs vLLM + evaluation deps in the venv
  --discover    Shows GPUs and basic info
  --run <P>     Generate data for phase P (2-5)
  --run-all     Runs all phases sequentially
  --slurm <P>   Submit Phase P to SLURM cluster
  --budget      Estimates budget using real benchmark data
  --pin <M>     Pins revision for model M from HuggingFace
  --bench <M>   Measures throughput of model M on GPU
  --info        Quick GPU info

Examples:
  $0 --setup                     # Setup environment
  $0 --pin Qwen/Qwen3-30B-A3B    # Pin revision
  $0 --run 2 --n 500             # Generate Phase 2 with N=500 samples
  $0 --slurm 3                   # Submit Phase 3 to SLURM

USAGE
        exit 1
        ;;
esac
