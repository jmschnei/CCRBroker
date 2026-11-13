# RUNBOOK COMPLETO — Ejecución con GPU de todas las Fases

**Repositorio:** `own_broker` (CCRBroker / routedata_kit)
**Fecha:** 2026-11-12
**Dependencia crítica:** todas las fases que generan inferencias requieren un entorno Linux con NVIDIA CUDA y vLLM ≥ 0.6.

---

## Índice

1. [Prerrequisitos y Preparación General](#1-prerequisitos-y-preparación-general)
2. [Fase 0 · Infraestructura — Medición de Throughput y Precios](#2-fase-0--infraestructura)
3. [Fase 1 · Reutilización y Evaluación Multi-Set](#3-fase-1--reutilización-y-evaluación-multiset)
4. [Fase 2 · ConfigRoute — Generación de Datasets con Coste/Esuerzo/Longitud](#4-fase-2--configroute)
5. [Fase 3 · DriftRoute — Deriva Temporal de Preferencias](#5-fase-3--driftroute)
6. [Fase 4 · AgentRoute — Enrutado Multi-Turno con Coste de Cambio](#6-fase-4--agentroute)
7. [Fase 5 · AdvRoute — Perturbaciones y Ataques](#7-fase-5--advroute)
8. [Receta SLURM (Schedulix / PBS / Slurm)](#8-receta-slurm)
9. [Checklist de Puertas de Paso](#9-checklist-de-puertas-de-paso)

---

## 1. Prerrequisitos y Preparación General

### 1.1 Hardware mínimo recomendado

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| GPU | 1× A100-80G (o H100) | 4× A100-80G (TP para modelos ≥70B) |
| CPU | 8 cores | 32+ cores |
| RAM | 32 GB | 128+ GB |
| Disco | 500 GB SSD (datos crudos ~200 GB) | 2 TB NVMe |
| Red | Para descargar modelos HF (sin GPU, CPU-only) | — |

### 1.2 Instalación del entorno

```bash
# En el servidor con NVIDIA/CUDA
cd /path/to/own_broker

# 1) Crear venv y deps base (CPU)
make install          # .venv + package editable + dev deps

# 2) Añadir deps GPU
source .venv/bin/activate
pip install -e '.[vllm]'        # instala vLLM + torch CUDA

# 3) (Opcional) deps de evaluación NLP
pip install -e '.[eval]'

# 4) Pre-commit hooks
pre-commit install

# 5) Verificar instalación base (sin GPU, sin red)
make test             # pytest: todas las pruebas sintéticas deben pasar
```

### 1.3 Descargar datos crudos (en máquina con CPU + red)

Antes de activar la GPU, descarga los datasets en una sesión normal:

```bash
source .venv/bin/activate
python -m routedata.download        # descarga RouterBench, SPROUT, LLMRouterBench a data/raw/

# Verificar que están presentes
ls data/raw/routerbench/
ls data/raw/sprout/
ls data/raw/llmrouterbench/extracted/bench-release/
```

### 1.4 Fijar revisiones HF en `pool_openweights.yaml`

**Obligatorio:** cada modelo debe tener su SHA de commit exacto de Hugging Face, no `PIN_ME`.

```python
# Ejecuta esto una vez para fijar revisiones (en máquina con red):
python << 'PYEOF'
import requests
for model in [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "zai-org/GLM-Z1-9B-0414",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
]:
    resp = requests.get(f"https://huggingface.co/api/models/{model}")
    if resp.status_code == 200:
        sha = resp.json().get("sha", "NO_SHA")
        print(f'{model}: {sha}')
PYEOF
```

Edita `configs/pool_openweights.yaml` con los SHAs obtenidos. **Ningún comando de Fase 1+ funcionará sin revisiones fijadas.**

---

## 2. Fase 0 · Infraestructura

### Objetivo
Medir throughput real por modelo, derivar precios por token y cerrar el gate de entrada a la Fase 1.

### Flujo

```bash
source .venv/bin/activate

# ============================================
# PASO 0: Medición individual de throughput
# Ejecuta un benchmark por modelo que vayas a usar.
# ============================================

# Modelo pequeño (1 GPU, TP=1)
make phase0-me MODEL="Qwen/Qwen3-0.6B" \
    GPU_USD_H="1.8" \
    GPU_PRICE_SOURCE="https://console.aws.amazon.com/batch/pricing/#instanceType=A100"

# Modelo mediano con thinking (1 GPU, TP=1)
make phase0-me MODEL="Qwen/Qwen3-8B" \
    GPU_USD_H="1.8" \
    GPU_PRICE_SOURCE="https://console.aws.amazon.com/batch/pricing/#instanceType=A100"

# Modelo grande (4 GPUs, TP=4 — se ajusta automáticamente por pool)
make phase0-me MODEL="meta-llama/Llama-3.1-70B-Instruct" \
    GPU_USD_H="6.0" \
    GPU_PRICE_SOURCE="https://console.aws.amazon.com/batch/pricing/#instanceType=A100"

# Modelo XL (8 GPUs, TP=8)
make phase0-me MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507" \
    GPU_USD_H="24.0" \
    GPU_PRICE_SOURCE="https://console.aws.amazon.com/batch/pricing/#instanceType=A100"

# Verificar resultados
cat outputs/phase0/throughput.jsonl
```

### Resultado esperado
Cada ejecución genera una línea JSON en `outputs/phase0/throughput.jsonl` con:
- `prefill_tokens_per_second`, `decode_tokens_per_second`
- `usd_per_million_input_tokens`, `usd_per_million_output_tokens`
- Meta-hardware (GPU name, count, vLLM version, torch version)

### Gate de paso Fase 0
> ✅ Cada modelo tiene al menos una medición válida con > 0 tok/s medidos.
> ⛔ Si algún benchmark falla, revisar compatibilidad vLLM/torch/CUDA del modelo.

---

## 3. Fase 1 · Reutilización y Evaluación Multi-Set

### Objetivo
Integrar los datasets públicos en el esquema unificado, ejecutar baselines locales y contrastar con auditoría oficial.

### Flujo completo

```bash
source .venv/bin/activate

# ============================================
# PASO 1: Regenerar dataset unificado
# ============================================
make phase1

# Validación estructural del unificado (manifiesto, IDs, FK, métricas)
make validate-unified

# Validación de los datos crudos
make validate
```

### Salida esperada
- `data/processed/unified/queries_combined.parquet` (~144.452 filas)
- `data/processed/unified/configs_combined.parquet` (~64 filas)
- `data/processed/unified/responses_combined.parquet` (~1,9M filas)
- `data/processed/unified/manifest.json`
- 0 duplicados, 0 FK huérfanas

### PASO 2: Auditoría del pool oficial (comprobación de fuentes)

```bash
make phase1-audit
# Genera outputs/phase1/protocol_audit.json y outputs/phase1/llmrouterbench_15datasets.json
```

### PASO 3: Ejecución de baselines locales × 5 semillas

```bash
make phase1-eval
# Ejecuta los 6 agentes baseline (random, cost, quality, latency, hybrid, heuristic)
# en los 15 datasets del pool oficial con las 5 semillas del PLAN
```

### PASO 4: Demostrar agentes de routing

```python
source .venv/bin/activate
python examples/routing_agents_demo.py     # demo sin datos (sin GPU)
python examples/load_unified_dataset.py    # estadísticas del unificado real
python examples/inspect_raw_datasets.py    # validación de data/raw/
```

### PASO 5: Evaluación en RouterArena `sub_10` (requiere GPU para generar)

Se usa el **evaluador oficial** de RouterArena (commit fijado en `generate/routerarena.py`,
`ARENA_COMMIT`), así que los números son comparables con su leaderboard. Ese evaluador solo
sabe llamar a APIs comerciales; nuestro puente genera las respuestas con vLLM sobre los
prompts que renderiza el propio RouterArena y le entrega el fichero de predicciones en su
formato. Los modelos locales se precian con su regla `local:` (`model_cost/local_pricing.yaml`).

Requisitos: `uv` (lo usa RouterArena; trae su propio Python 3.11/3.12) y red para el setup.

```bash
source .venv/bin/activate

# 1) Una vez (CPU + red): clona RouterArena en external/, `uv sync` y prepara el dataset
make phase1-arena-setup

# 2) GPU: respuestas de los 9 modelos del pool a los 809 prompts (reanudable, en
#    outputs/routerarena/cache/). Alternativa sin cargar vLLM en proceso:
#    python scripts/run_routerarena.py generate --backend openai --endpoints configs/agent_endpoints.json
make phase1-arena-generate            # BACKEND=mock para probar el flujo sin GPU

# 3) CPU: cada candidato como router de un solo modelo → accuracy, coste/1K y Arena score
make phase1-arena-candidates          # → outputs/routerarena/candidates_sub_10.json

# 4) CPU: un router propio a partir de sus decisiones {global_index: modelo}
python scripts/run_routerarena.py evaluate --router mi-router --choices decisiones.json
#    → outputs/routerarena/mi-router_sub_10.json (incluye métricas de optimalidad)
```

> ⚠️ Reglas de RouterArena: es **solo de evaluación**. El router no puede entrenarse ni
> ajustarse con sus datos y debe decidir con el contenido de la consulta, no con el dataset de
> origen ni con la plantilla del prompt (`routedata.adapters.base.assert_trainable` bloquea lo primero).

### Gate de paso Fase 1
> ✅ Dataset unificado validado sin duplicados ni FK huérfanas.
> ✅ Auditoría confirma los 20 modelos del pool en los 15 datasets.
> ✅ Baselines locales reproducidos (macro: random≈0,478; best-single≈0,718; oracle_query≈0,916).
> ⛔ Faltan: reproducción de ≥5 métodos publicados y evaluación completa en RouterArena `sub_10`.

---

## 4. Fase 2 · ConfigRoute

### Objetivo
Generar un dataset nuevo con la rejilla configuración × esfuerzo × longitud, etiquetado por verificador + juez LLM + auditoría humana.

### Preparación de pool para piloto

En `configs/pool_openweights.yaml`, selecciona **4 modelos para el piloto**:
```yaml
models:
  Qwen/Qwen3-0.6B:        {revision: "SHA_REAL", thinking: true, size_b: 0.6, tp: 1}     # pequeño
  Qwen/Qwen3-8B:          {revision: "SHA_REAL", thinking: true, size_b: 8, tp: 1}       # mediano (thinking)
  google/gemma-2-9b-it:   {revision: "SHA_REAL", thinking: false, size_b: 9, tp: 1}      # mediano
  meta-llama/Llama-3.1-70B-Instruct: {revision: "SHA_REAL", thinking: false, size_b: 70, tp: 4}  # grande (si cabe)
```

### PASO 1: Construir queries del piloto

```bash
source .venv/bin/activate

python -m generate.run_phase2 build-queries \
    --cfg configs/configroute_pilot.yaml \
    --out data/queries \
    --open-ended data/queries/open_ended.jsonl   # (opcional, con licencias verificadas)
```

**Pre-chequeo:** compara columnas antes de generar:
```bash
python -m generate.build_queries inspect mmlu_pro
python -m generate.build_queries inspect math500
python -m generate.build_queries inspect gpqa_diamond
python -m generate.build_queries inspect humaneval
```

### PASO 2: Ejecutar piloto (~1.000 consultas)

#### Sin SLURM (GPU dedicada):
```bash
source .venv/bin/activate

python -m generate.run_phase2 pilot \
    --pool configs/pool_openweights.yaml \
    --cfg configs/configroute_pilot.yaml \
    --queries data/queries \
    --out data/pilot \
    --backend vllm        # usa los modelos del pool en pool_openweights.yaml
```

#### Con SLURM (batch):
```bash
# submit_pilot.sh
#!/bin/bash
#SBATCH --job-name=phase2-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err

module load cuda/12.x
source /path/to/own_broker/.venv/bin/activate

cd /path/to/own_broker
python -m generate.run_phase2 pilot \
    --pool configs/pool_openweights.yaml \
    --cfg configs/configroute_pilot.yaml \
    --queries data/queries \
    --out data/pilot \
    --backend vllm
```

**Qué vigilar en el log:** truncados masivos → subir `max_model_len`; respuestas sin «Final answer:» → revisar prompt template antes de culpar al modelo.

### PASO 3: Etiquetado con juez LLM + auditoría humana

```bash
source .venv/bin/activate

# Juez (usa modelos en pool.judges)
python -m generate.run_phase2 judge \
    --run-dir data/pilot \
    --pool configs/pool_openweights.yaml \
    --backend vllm

# Exportar hoja ciega para 150 ítems humanos
python -m generate.run_phase2 audit-export \
    --run-dir data/pilot \
    --n 150
# Resultado: data/pilot/audit/audit_sheet.csv (ciega)

# Repartir la hoja entre 2 anotadores, que rellenan human_score_a/b (0–1)

# Importar respuestas humanas
python -m generate.run_phase2 audit-import \
    --run-dir data/pilot \
    --sheet data/pilot/audit/audit_sheet_filled.csv
```

### PASO 4: Generar informe y evaluar puertas de paso

```bash
source .venv/bin/activate

python -m generate.run_phase2 report \
    --run-dir data/pilot \
    --pool configs/pool_openweights.yaml \
    --cfg configs/configroute_pilot.yaml \
    --n-full 8000
# Genera: data/pilot/REPORT.md (aciertos, tokens, truncados, tok/s, GPU-horas proyectadas)
```

### Puertas de Fase 2 (no pasar sin cumplirlas)

| Puerta | Umbral | Cómo verificar |
|--------|--------|----------------|
| Juez vs humano: Pearson ≥ 0,8 y κ ≥ 0,7 | `judge_vs_human.json` en report | Si falla → revisar instructivo de anotación o jueces del pool |
| Cumplimiento longitud ≥ 80% (modelos grandes) | En REPORT.md `big_models_compliance` | Si falla → ajustar presupuesto tokens |
| Coste proyectado dentro del presupuesto | En REPORT.md GPU-horas | Si no cabe → reducir configs/modelos, no el rigor |

### PASO 5: Escala completa (tras superar puertas)

```bash
# Copiar config de piloto a escala
cp configs/configroute_pilot.yaml configs/configroute_scale.yaml
# Editar configscale.yaml: subir n_queries_total a 8000-20000, añadir think_high y budgets

python -m generate.run_phase2 pilot \
    --pool configs/pool_openweights.yaml \
    --cfg configs/configroute_scale.yaml \
    --queries data/queries \
    --out data/configroute \
    --backend vllm

# Judge + audit_n=500 para escala completa
python -m generate.run_phase2 judge --run-dir data/configroute --pool configs/pool_openweights.yaml --backend vllm
python -m generate.run_phase2 audit-export --run-dir data/configroute --n 500
python -m generate.run_phase2 report --run-dir data/configroute --pool configs/pool_openweights.yaml --cfg configs/configroute_scale.yaml --n-full 8000
```

---

## 5. Fase 3 · DriftRoute

### Objetivo
Medir deriva temporal de preferencias con snapshots mensuales sobre 3.000 consultas, comparando cambios en el pool de modelos y precios.

### Preparación

En `configs/driftroute.yaml` (ya incluido), verificar:
```yaml
subset_queries: 3000        # no cambiar entre snapshots
verifiable_only: true       # true = evita juez LLM para empezar
test_frac: 0.5              # partición fija por cluster — NO regenerar nunca
fractions: [0.0, 0.005, 0.01, 0.02, 0.05, 0.2, 1.0]  # tráfico de calibración E4
```

### PASO 1: Congelar subconjunto (una sola vez)

```bash
source .venv/bin/activate

python -m generate.run_phase3 init \
    --cfg configs/driftroute.yaml \
    --drift-dir data/drift \
    --queries data/processed/unified/queries_combined.parquet \
    --exclude-ids data/private_10pct.jsonl   # (opcional: partición privada del 10%)
```

Resultado: `data/drift/subset/{queries.*, split.json, subset_manifest.json}` — **NO CAMBIAR ESTO NUNCA**.

### PASO 2: Snapshots "placebo" (sin eventos) — para medir ruido base

#### Sin SLURM:
```bash
source .venv/bin/activate

# Snapshot s00
python -m generate.run_phase3 snapshot \
    --drift-dir data/drift \
    --snapshot s00 \
    --pool configs/pool_openweights.yaml \
    --date 2026-11-15 \
    --backend vllm

# Snapshot s01 (mismo pool, otro día — mide ruido de determinismo)
python -m generate.run_phase3 snapshot \
    --drift-dir data/drift \
    --snapshot s01 \
    --pool configs/pool_openweights.yaml \
    --date 2026-11-16 \
    --backend vllm

# Verificar comparabilidad y ruido base
python -m generate.run_phase3 check --drift-dir data/drift
python -m generate.run_phase3 report --drift-dir data/drift
# Los "flips" entre s00 y s01 son el suelo de ruido — eventos posteriores deben superarlo
```

#### Con SLURM:
```bash
# submit_drift_snapshot.sh
#!/bin/bash
#SBATCH --job-name=drift-s00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:8          # si el pool usa varios modelos con TP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=logs/drift_sXX_%j.out
#SBATCH --error=logs/drift_sXX_%j.err

module load cuda/12.x
source /path/to/own_broker/.venv/bin/activate

cd /path/to/own_broker
python -m generate.run_phase3 snapshot \
    --drift-dir data/drift \
    --snapshot s00 \
    --pool configs/pool_openweights.yaml \
    --date 2026-11-15 \
    --backend vllm
```

### PASO 3: Registrar eventos y snapshots con cambios reales

#### Ejemplo: evento de nuevo modelo (snapshot s02)

```bash
source .venv/bin/activate

# Registrar eventos en pool_s02.yaml
python -m generate.run_phase3 new-pool \
    --base-pool configs/pool_openweights.yaml \
    --out data/drift/pool_s02.yaml \
    --snapshot s02 \
    --add "new-model-org/NewModel-14B=<SHA_COMMIT>,14,1,1" \
    --retire "google/gemma-2-9b-it" \
    --revision "Qwen/Qwen3-8B=<SHA_NUEVO>"

# Ejecutar snapshot con pool actualizado
python -m generate.run_phase3 snapshot \
    --drift-dir data/drift \
    --snapshot s02 \
    --pool data/drift/pool_s02.yaml \
    --date 2026-12-15 \
    --backend vllm

# Lote nuevo de consultas (opcional, solo para test)
python -m generate.run_phase3 snapshot \
    --drift-dir data/drift \
    --snapshot s03 \
    --pool data/drift/pool_s02.yaml \
    --date 2027-01-15 \
    --backend vllm \
    --new-batch-file data/batches/new_queries.jsonl \
    --batch-id b1
```

### PASO 4: Evaluación E4 (recomendado tras ≥3 snapshots)

```bash
source .venv/bin/activate

python -m generate.run_phase3 eval --drift-dir data/drift
# Genera: data/drift/e4_results.csv y data/drift/E4_REPORT.md
# Compara static, refit(f), profile_online(f) routers por evento
```

### PASO 5: Verificar y reportar periódicamente

```bash
source .venv/bin/activate

python -m generate.run_phase3 check --drift-dir data/drift   # comparabilidad entre snapshots
python -m generate.run_phase3 report --drift-dir data/drift   # deriva por config con McNemar+Holm
```

### Puerta de paso Fase 3
> ✅ ≥ 6 snapshots antes de afirmar tendencias.
> ✅ ≥ 3 eventos reales de pool/precio/revisión (o replays sobre versiones antiguas).
> ✅ Suelo de ruido medido y reportado.
> ⛔ Faltan: IC por evento (bootstrap pendiente), ejecución real con vLLM/GPU.

---

## 6. Fase 4 · AgentRoute

### Objetivo
Comparar enrutado por paso (con coste de cambio) frente a enrutado por tarea en tareas multi-turno, demostrando H5.

### Preparación

1. Asegurar que `prices.json` viene de throughput medido (Fase 2/3), no de inferencia secuencial de agentes.
2. Implementar al menos un entorno real: **MultiHopQAEnv** es el más viable (busqueda local BM25).

### PASO 0: Servidores vLLM (uno por modelo)

```bash
# Ejecutar en terminal separada, cada uno en su puerto:
vllm serve Qwen/Qwen3-0.6B --port 8001 --enable-prefix-caching --tensor-parallel-size 1
vllm serve google/gemma-2-9b-it --port 8002 --enable-prefix-caching --tensor-parallel-size 1
vllm serve meta-llama/Llama-3.1-70B-Instruct --port 8003 --enable-prefix-caching --tensor-parallel-size 4

# Crear endpoints.json con las URLs:
cat > agent_endpoints.json << 'EOF'
{
  "Qwen/Qwen3-0.6B": "http://localhost:8001",
  "google/gemma-2-9b-it": "http://localhost:8002",
  "meta-llama/Llama-3.1-70B-Instruct": "http://localhost:8003"
}
EOF
```

### PASO 1: Inicializar tareas y partición por tarea

```bash
source .venv/bin/activate

# Para entorno sintético (ensayo en seco):
python -m generate.run_phase4 init \
    --data-dir data/agent \
    --env synthetic \
    --env-kw '{"horizon": 8, "n_hard": 3, "obs_chars": 2000}' \
    --n-tasks 200 \
    --test-frac 0.4
```

**Pre-chequeo (obligatorio antes de escalar):**
```python
# Determinismo: AG.check_deterministic(env_factory, seed, acciones) ≥ 0.95
# Validez formato: ≥ 80% acciones válidas por modelo en ~20 episodios
# Headroom: ver si hay margen para enrutar con "analyze"
```

### PASO 2: Derivar precios desde la Fase 2/3

```bash
source .venv/bin/activate

python -m generate.run_phase4 make-prices \
    --data-dir data/agent \
    --configs data/drift/s00/configs_combined.parquet   # configs de Fase 3, o las de pilot
```

### PASO 3: Recoger episodios fijos y mixtos (reanudable)

#### Sin SLURM:
```bash
source .venv/bin/activate

python -m generate.run_phase4 collect \
    --data-dir data/agent \
    --endpoints agent_endpoints.json \
    --models "Qwen/Qwen3-0.6B,google/gemma-2-9b-it,meta-llama/Llama-3.1-70B-Instruct" \
    --workers 8
```

#### Con SLURM:
```bash
# submit_agent_collect.sh
#!/bin/bash
#SBATCH --job-name=agent-collect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00

module load cuda/12.x torch  # que incluya openai compatible con backend
source /path/to/own_broker/.venv/bin/activate

cd /path/to/own_broker
python -m generate.run_phase4 collect \
    --data-dir data/agent \
    --endpoints agent_endpoints.json \
    --models "Qwen/Qwen3-0.6B,google/gemma-2-9b-it,meta-llama/Llama-3.1-70B-Instruct" \
    --workers 8
```

### PASO 4: Etiquetas contrafactuales por estado (train)

#### Modo local (barato — muestra de un solo paso):
```bash
source .venv/bin/activate

python -m generate.run_phase4 stepdata \
    --data-dir data/agent \
    --endpoints agent_endpoints.json \
    --models "Qwen/Qwen3-0.6B,google/gemma-2-9b-it,meta-llama/Llama-3.1-70B-Instruct" \
    --reference-model meta-llama/Llama-3.1-70B-Instruct \
    --n-samples 3 \
    --max-trajectories 100 \
    --mode local \
    --cache-mode prefix \
    --eps 0.4
```

#### Modo rollout (caro — captura consecuencias a largo plazo):
```bash
# Se multiplica el coste: cada muestra es una trayectoria completa al modelo de referencia
python -m generate.run_phase4 stepdata \
    --data-dir data/agent \
    --endpoints agent_endpoints.json \
    --models "Qwen/Qwen3-0.6B,google/gemma-2-9b-it,meta-llama/Llama-3.1-70B-Instruct" \
    --reference-model meta-llama/Llama-3.1-70B-Instruct \
    --n-samples 3 \
    --max-trajectories 100 \
    --mode rollout \
    --cache-mode prefix
```

### PASO 5: Entrenar routers, elegir λ en validación y evaluar EN LÍNEA sobre test

```bash
source .venv/bin/activate

python -m generate.run_phase4 evaluate \
    --data-dir data/agent \
    --endpoints agent_endpoints.json \
    --models "Qwen/Qwen3-0.6B,google/gemma-2-9b-it,meta-llama/Llama-3.1-70B-Instruct" \
    --lams "10,30,100,300,1000" \
    --cache-mode prefix \
    --retention keep
```

### PASO 6: Generar informe de H5

```bash
source .venv/bin/activate

python -m generate.run_phase4 analyze \
    --data-dir data/agent \
    --models "Qwen/Qwen3-0.6B,google/gemma-2-9b-it,meta-llama/Llama-3.1-70B-Instruct"
# Genera: data/agent/AGENT_REPORT.md
```

### PASO 7: Simulación completa (ensayo en seco con MockBackend)

```bash
source .venv/bin/activate

python -m generate.run_phase4 simulate \
    --n-tasks 60
# Ejecuta init → collect → stepdata → evaluate → analyze con entorno sintético + mock
# Resultado rápido sin GPU pero solo ilustrativo
```

### Puerta de paso Fase 4
> ✅ Determinismo ≥ 0,95 (si se usan contrafactuales/clones).
> ✅ Tasa acciones válidas ≥ 80%.
> ✅ Headroom medible.
> ✅ Modelo de caché validado.
> ⛔ Faltan: adaptadores reales de τ²-Bench, ScienceWorld, SWE-bench, QA multi-salto.

---

## 7. Fase 5 · AdvRoute

### Objetivo
Medir el impacto de perturbaciones benignas y ataques black-box sobre la calidad y coste del routing.

### Estado actual
Código estructura en `generate/advroute.py` pero **no hay runbook PHASE5 ni CLI ejecutable**.

### PASO 1: Ejecutar simulador (ya funciona)

```bash
source .venv/bin/activate

# Configurar objetivo y perturbaciones en configs/advroute.yaml
# benign: [typos, whitespace, pad, option_shuffle, paraphrase, translate_es_de_fr]
# attacks: gadget_black_box {length: 10, iters: 200}, adaptive_vs_detector: true
# objectives: [inflate_cost, degrade_quality]

python << 'PYEOF'
from generate import advroute as AVR
# Simular dataset de respuestas + aplicar perturbaciones beningnas
# y ataques gadget_black_box sobre el pool
PYEOF
```

### PASO 2: Implementar la CLI (pendiente)

La estructura está en `advroute.py`. Se necesita completar:
- Funciones de perturbación por tipo (`typos`, `whitespace`, `pad`, `option_shuffle`, `paraphrase`)
- `gadget_black_box`: itera `length` tokens × `iters` iteraciones para maximizar coste o degradar calidad
- Detector adaptativo y evaluación de consistencia
- CLI análoga a las otras fases (`init`, `perturb`, `attack`, `detect`, `evaluate`)

---

## 8. Receta SLURM (Schedulix / PBS / Slurm)

### 8.1 Solicitud de recursos por fase

| Fase | GPU(s) | TP máximo | Tiempo estimado | Notas |
|------|--------|-----------|-----------------|-------|
| 0 | 1× A100 | 1–8 (por modelo) | 30 min/modelo | Medición individual |
| 1 | 1× A100 | 1 | 2–4 h | Auditoría + baselines locales |
| 2 piloto | 4× A100 | 4 (70B si se incluye) | 6–24 h | ~1.000 queries × configs |
| 2 escala | 8× A100 | 8 | 2–5 días | ~20.000 queries × 60 configs |
| 3 (por snapshot) | 4–8× A100 | 8 | 1–3 días/snapshot | 3.000 queries × ~4 configs × N modelos |
| 4 collect | 4–8× A100 | 4–8 | 1–3 días | 200 tareas × episodios fijos/mixtos |
| 4 stepdata | 4× A100 | 4 | 2–4 días | rollouts múltiples por estado |

### 8.2 Script plantilla SLURM genérico

```bash
#!/bin/bash
#SBATCH --job-name=PHASE_TAG
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:NUM_GPUS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=DAYS:00:00
#SBATCH --output=logs/phase_TAG_%j.out
#SBATCH --error=logs/phase_TAG_%j.err

# ===== CONFIGURAR SEGÚN TU CLÚSTER =====
module load cuda/12.x                    # o whatever your cluster uses
export CUDA_VISIBLE_DEVICES=$SLURM_GPUS_ID   # o $CUDA_VISIBLE_DEVICES

source /path/to/own_broker/.venv/bin/activate
cd /path/to/own_broker

# ===== COMANDO DE FASE (reemplazar según fase) =====
python -m generate.run_phaseX ...

# O para make targets:
# make phase0-me MODEL="..." GPU_USD_H="..." GPU_PRICE_SOURCE="..."
```

### 8.3 Script de cola masiva (Slurm array jobs)

Para ejecutar benchmark de throughput en varios modelos en paralelo:

```bash
#!/bin/bash
#SBATCH --job-name=bench-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                # un GPU por tarea del array
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-8                       # 9 modelos (0..8)

MODEL_LIST="Qwen/Qwen3-0.6B Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b-it zai-org/GLM-Z1-9B-0414 Qwen/Qwen2.5-Coder-7B-Instruct deepseek-ai/DeepSeek-R1-0528-Qwen3-8B meta-llama/Llama-3.1-70B-Instruct Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL=$(echo "$MODEL_LIST" | cut -d' ' -f$(($SLURM_ARRAY_TASK_ID + 1)))

source /path/to/own_broker/.venv/bin/activate
cd /path/to/own_broker

make phase0-me MODEL="$MODEL" \
    GPU_USD_H="1.8" \
    GPU_PRICE_SOURCE="aws-batch-a100" \
    2>&1 | tee outputs/phase0/bench_${MODEL##*/}.log
```

---

## 9. Checklist de Puertas de Paso

### Resumen global por fase

| Fase | Gate | Estado actual | Qué ejecutar para cerrar |
|------|------|---------------|-------------------------|
| **Fase 0** | `tok/s` medidos + precios derivados en `pricing.py` | ⛔ Sin CUDA | `make phase0-me MODEL=... GPU_USD_H=... GPU_PRICE_SOURCE=...` para cada modelo del pool |
| **Fase 1** | Auditoría oficial ✅ + baselines reproducidas | ✅ Estructura ✅ Resultados preliminares | `make phase1`, `make validate-unified`, `make phase1-eval`, evaluar en RouterArena `sub_10` con GPU |
| **Fase 1 (pend.)** | Reproducir ≥5 métodos publicados de LLMRouterBench | ⛔ Solo baselines locales propias | Implementar los 5 routers originales (no solo las baselines del kit) |
| **Fase 2 piloto** | Juez vs humano: Pearson≥0,8 / κ≥0,7 + longitud≥80% | ⛔ Sin datos reales | `run_phase2 build-queries → pilot → judge → audit-export/import → report` con vLLM real |
| **Fase 2 escala** | Coste proyectado dentro de presupuesto | ⛔ Proyección solo (precios=0) | Reducir rejilla si el coste es prohibitivo; escalar a 8.000–20.000 queries con precios derivados reales |
| **Fase 3** | ≥6 snapshots + ≥3 eventos reales | ⛔ Sin snapshots | `run_phase3 init → snapshot s00/s01/placebo → check/report → new-pool → snapshot(s) con eventos` |
| **Fase 3 E4** | Router comparison (static vs refit vs profile_online) | ⛔ Pendiente de ≥3 snapshots | `run_phase3 eval` tras tener ≥3 snapshots reales |
| **Fase 4** | Determinismo≥0,95 + acciones válidas≥80% + headroom | ⛔ Sin entornos reales | Implementar MultiHopQAEnv (primero), ejecutar pre-chequeos; luego escalar con vLLM |
| **Fase 4 H5** | step_router_penalty > task_router en cache_mode=prefix | ⛔ Solo simulador | Ejecutar toda la tubería: `init → make-prices → collect → stepdata → evaluate → analyze` con servidores vLLM reales |
| **Fase 5** | CLI de perturbaciones + detector implementado | ⛔ Solo estructura de código | Completar `advroute.py` con funciones de perturbación, ataque y detección; escribir PHASE5_RUNBOOK.md |

### Orden recomendado de ejecución (dependencias entre fases)

```
Fase 0 (throughput/pricing)
    ↓
Fase 1a (dataset unificado + validación) → Fase 1b (baselines locales)
    ↓ (en paralelo con Fase 1b, si hay GPU suficiente)
Fase 2 piloto (ConfigRoute) ──────────────→ Fase 3 (DriftRoute, primer snapshot)
    ↓                                      ↓
Fase 2 escala (si gates superados)        Fase 3 snapshots mensuales + E4
    ↓
Fase 4 (AgentRoute) ──────────────────────→ Fase 5 (AdvRoute, independiente)
```

### Dependencias clave entre fases

| Fase depende de | Por qué |
|-----------------|---------|
| Fase 1 ← Fase 0 | Precios derivados del throughput son necesarios para métricas de coste en la evaluación |
| Fase 2 ← Fase 0 | Precios reales por tok → precios en configs |
| Fase 3 ← Fase 0 | Mismo requisito: precios correctos para reprice_configs |
| Fase 3 ← Fase 2 (parcial) | Los prompts/efforts de la Fase 2 deben congelarse antes del primer snapshot DriftRoute |
| Fase 4 ← Fase 2/3 | Precios derivados de throughput medido en lotes, no de inferencia secuencial de agentes |
| Fase 5 | Independiente — puede ir en paralelo con cualquier fase previa |

---

## A. Comandos rápidos de referencia

```bash
# Instalación GPU
make install && pip install -e '.[vllm]'

# Fase 0
make phase0-me MODEL="Qwen/Qwen3-0.6B" GPU_USD_H="1.8" GPU_PRICE_SOURCE="aws-a100"

# Fase 1
make phase1 && make validate-unified && make phase1-eval && make phase1-audit

# Fase 2 (pilot)
python -m generate.run_phase2 build-queries --cfg configs/configroute_pilot.yaml --out data/queries
python -m generate.run_phase2 pilot --pool configs/pool_openweights.yaml --cfg configs/configroute_pilot.yaml --queries data/queries --out data/pilot --backend vllm
python -m generate.run_phase2 judge --run-dir data/pilot --pool configs/pool_openweights.yaml --backend vllm
python -m generate.run_phase2 report --run-dir data/pilot --pool configs/pool_openweights.yaml --cfg configs/configroute_pilot.yaml --n-full 8000

# Fase 3
python -m generate.run_phase3 init --cfg configs/driftroute.yaml --drift-dir data/drift --queries data/processed/unified/queries_combined.parquet
python -m generate.run_phase3 snapshot --drift-dir data/drift --snapshot s00 --pool configs/pool_openweights.yaml --date 2026-11-15 --backend vllm
python -m generate.run_phase3 check --drift-dir data/drift && python -m generate.run_phase3 report --drift-dir data/drift

# Fase 4 (simulación rápida)
python -m generate.run_phase4 simulate --n-tasks 60
```
