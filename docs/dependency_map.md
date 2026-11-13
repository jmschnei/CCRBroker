# 🔗 Mapa de dependencias cruzadas

Este documento documenta todas las relaciones entre módulos del proyecto.

---

## 1. Dependencias core (routedata/)

### `schema.py` — Modelo unificado (dependencia raíz)
```python
# Usado por TODOS los módulos que manejan datos
from routedata.schema import Query, Config, Response, save_table, load_table
```

**Dependencias directas:**
- ✅ `pipeline.py` — ingiere datasets crudos en schemas
- ✅ `metrics.py` — recibe DataFrames de queries/configs/responses
- ✅ `baselines.py` — compara con baselines sobre responses
- ✅ `conformal.py` — valida cobertura sobre predictions
- ✅ `download.py` — descarga → normaliza a schema unificado

### `pipeline.py` — Ingestión y unificación
```python
from routedata.pipeline import load_dataset, save_unified
# O usa: from routedata.adapters import RouterBenchAdapter, ...
```

**Dependencias directas:**
- ✅ `download.py` — descarga datasets crudos que pipeline normaliza
- ✅ `validate.py` — valida lo que pipeline produce (unificado)
- ✅ `routedata/adapters/*` — cada adaptador genera formato intermedio para pipeline

**Dependientes de pipeline:**
- 🔹 `regenerate_combined.py` — re-ejecuta pipeline completo
- 🔹 `tests/test_pipeline.py` — tests de pipeline

### `metrics.py` — Métricas de evaluación
```python
from routedata.metrics import gain_at_b, gap_at_o, aiq, audc, ece_brier
```

**Dependencias directas:**
- ✅ `baselines.py` — usa métricas para comparar agents
- ✅ `conformal.py` — calcula coverage sobre predictions de routers

**Usado por scripts:**
- 🔹 `scripts/run_phase1_multidataset.py` — evalúa baselines con gain_at_b, gap_at_o
- 🔹 `tests/test_synthetic.py` — verifica métricas en datos sintéticos
- 🔹 `examples/load_unified_dataset.py` — muestra estadísticas

### `stats.py` — Estadística pura (sin dependencias externas)
```python
from routedata.stats import bootstrap_ci, mcnemar_exact, holm_correction
```

**Usado por:**
- 🔹 `routedata/driftroute.py` — McNemar + Holm en drift report
- 🔹 `routedata/metrics.py` — CI bootstrap para todas las métricas
- 🔹 `generate/run_phase4.py` — análisis de significancia en H5

### `conformal.py` — Control de riesgo conformal (CRC)
```python
from routedate.conformal import conformal_prediction, coverage_guarantee
```

**Dependencias directas:**
- ✅ `pricing.py` — no hay dependencia directa pero usa mismo concepto de garantía estadística

**Usado por:**
- 🔹 E7/RQ7 del proyecto — garantiza cobertura ≥ α para decisiones de routing bajo riesgo

### `pricing.py` — Cálculo de costes GPU
```python
from routedata.pricing import derive_selfhost_prices, scale_prices, reprice_configs
```

**Dependencias directas:**
- ✅ Ninguna (depende solo de cálculos matemáticos puros)

**Usado por:**
- 🔹 `scripts/benchmark_vllm.py` — deriva precios desde throughput medido
- 🔹 `generate/run_phase3.py` — reprice_configs en cada snapshot DriftRoute
- 🔹 `generate/run_phase4.py` — make-prices para AgentRoute

### `splits.py` — Estrategias de partición (sin dependencias)
```python
from routedata.splits import split_iid, split_cluster, split_temporal
```

**Usado por:**
- 🔹 `generate/configroute.py` — sample_queries usa stratified splitting
- 🔹 `generate/driftroute.py` — freeze_subset + split_train_test
- 🔹 `scripts/run_phase1_multidataset.py` — aplica splits LLMRouterBench (seeds)

---

## 2. Dependencias generation framework (generate/)

### `backends.py` — Abstracción de inferencia (dependencia raíz para fases 2–5)
```python
from generate.backends import VLLMBackend, MockBackend, GenRequest, GenResult
```

**Dependencias directas:**
- ✅ `torch` — GPU CUDA para VLLMBackend
- ✅ `httpx` / `openai` — cliente HTTP para endpoints de modelos

**Usado por TODAS las fases que generan inferencias:**
- 🔹 `generate/agentroute.py` — run_episode() usa backends
- 🔹 `generate/driftroute.py` — snapshot phase usa VLLMBackend real
- 🔹 `generate/configroute.py` — run() usa backends para generación
- 🔹 `generate/judge.py` — LLM judge usa backends
- 🔹 `scripts/benchmark_vllm.py` — benchmark directo de throughput

### `configroute.py` — Core Fase 2 (depende de schema + backends)
```python
from generate.configroute import sample_queries, build_config_grid, run
```

**Dependencias directas:**
- ✅ `routedata/schema.py` — Query/Config/Response pydantic models
- ✅ `generate/backends.py` — inferencia de modelos
- ✅ `generate/prompts.py` — templates para prompts multi-dominio
- ✅ `generate/efforts.py` — effort/budget forcing logic
- ✅ `generate/verifiers.py` — verification programática por tarea

**Usado por:**
- 🔹 `generate/run_phase2.py` — CLI orchestrator Fase 2 (build-queries, pilot)
- 🔹 `generate/driftroute.py` — config_grid para DriftRoute
- 🔹 `tests/test_phase1_multidataset.py` — sample_queries verification

### `driftroute.py` — Core Fase 3 (depende de schema + pricing)
```python
from generate.driftroute import freeze_subset, split_train_test, drift_report
```

**Dependencias directas:**
- ✅ `routedata/schema.py` — Query/Config pydantic models
- ✅ `routedata/pricing.py` — reprice_configs por throughput medido
- ✅ `routedata/drift.py` — SnapshotManifest + drift detection utilities
- ✅ `routedata/stats.py` — mcnemar_exact, holm_correction en drift_report
- ✅ `generate/configroute.py` — build_config_grid (reutiliza grid generation)

**Usado por:**
- 🔹 `generate/run_phase3.py` — CLI orchestrator Fase 3 (init, snapshot, check, report, eval)
- 🔹 `tests/test_phase3.py` — synthetic + MockBackend tests

### `agentroute.py` — Core Fase 4 (depende de schema + agent utilities)
```python
from generate.agentroute import run_episode, collect_step_utilities, TaskRouter, StepRouter
```

**Dependencias directas:**
- ✅ `routedata/schema.py` — Query/Config pydantic models
- ✅ `routedata/agent_routing.py` — utilidades para episodes y routing decisions
- ✅ `generate/backends.py` — VLLMBackend real o MockBackend
- ✅ `generate/envs/` — entornos (synthetic + adapters_template)

**Usado por:**
- 🔹 `generate/run_phase4.py` — CLI orchestrator Fase 4 (init, collect, stepdata, evaluate, analyze)
- 🔹 `tests/test_phase4.py` — MockBackend + synthetic env tests

### `advroute.py` — Core Fase 5 (independiente, estructura básica)
```python
from generate.advroute import perturb_benign, gadget_black_box
```

**Dependencias directas:**
- ✅ `routedata/metrics.py` — mide efecto de perturbaciones en métricas

**Usado por:**
- 🔹 No hay orquestador CLI aún (Fase 5 no implementada completamente)
- 🔹 Tests de validación sintéticos

---

## 3. Scripts standalone y examples

### `scripts/benchmark_vllm.py` — Medición GPU
```python
# Depende de: generate.backends.VLLMBackend + routedata.pricing.derive_selfhost_prices
```
→ Usa VLLMBackend directamente para medir throughput por modelo
→ Escribe resultados a `outputs/phase0/throughput.jsonl`

### `scripts/run_phase1.py` — Ejecución directa Fase 1
```python
# Depende de: routedata.metrics + baselines + stats
```
→ Corre baselines locales (6 agents) sobre datos unificados
→ Genera outputs/phase1/* con resultados por dataset

### `scripts/run_phase1_multidataset.py` — Auditoría multi-dataset
```python
# Depende de: LLMRouterBench official pool + routedata.downloader
```
→ Audita pool oficial 20 modelos × 15 datasets
→ Verifica reproducibilidad vs. resultados published

### `examples/routing_agents_demo.py` — Demo agentes routing
```python
from routedata.agents import create_agent, Candidate
```
→ Sin datos reales (solo muestra API de agents)
→ Útil para testing y documentación de interfaz

### `examples/load_unified_dataset.py` — Carga dataset unificado
```python
from pandas import read_parquet
# Loads data/processed/unified/* from pipeline output
```
→ Estadísticas básicas + integridad del dataset procesado
→ Ejemplo de uso real de schema unificado

---

## 4. Diagrama completo de dependencias

```
schema.py (Pydantic models) ←─ dependency root
    │
    ├──► pipeline.py (ingestion, unification)
    │       └──► adapters/* (RouterBench, SPROUT, etc.)
    │
    ├──► metrics.py (Gain@b, Gap@O, AIQ...)
    │       ├──► baselines.py (6 agents baseline)
    │       └──► conformal.py (CRC guarantee)
    │
    ├──► splits.py (iid, cluster, temporal)
    │       └──► used by configroute + driftroute
    │
    ├──► stats.py (bootstrap, McNemar, Holm)
    │       └──► drift_report + all metric CIs
    │
    ├──► pricing.py (GPU costs from tok/s)
    │       └──► reprice_configs in Phase3 snapshots
    │
    └──► download.py (datasets crudos → normalized)

generate/backends.py (VLLM/Mock inference) ←─ dependency root for phases 2-5
    ├──► configroute.py (Fase 2 core: sample_queries, build_grid, run)
    │       ├──► prompts.py (templates multi-dominio)
    │       ├──► efforts.py (budget forcing)
    │       └──► verifiers.py (programatic verification)
    │
    ├──► driftroute.py (Fase 3 core: freeze, split, drift)
    │       └───► reprice_configs from pricing.py
    │
    ├──► agentroute.py (Fase 4 core: episodes, step utilities, routers)
    │       ├──► envs/synthetic.py (test environment)
    │       └──► envs/adapters_template.py (τ², ScienceWorld, etc.)
    │
    └──► advroute.py (Fase 5 structure: perturbations + attacks)

run_phase2.py ← configroute + build_queries + phase2 + judge + verifiers
run_phase3.py ← driftroute + configroute + schema + pricing
run_phase4.py ← agentroute + envs/* + schema + stats
```

---

## 5. Resumen de imports circulares (none detected)

El código está diseñado para evitar circular dependencies:

| Patrón | Implementación |
|--------|----------------|
| **Core independiente** | `routedata/*` puede usarse sin `generate/` |
| **Phases como consumers** | `generate/*` importa de `routedata`, no al revés |
| **Backends abstractos** | `backends.py` define interfaces, implementación real es runtime |
| **Adapters decoupled** | Cada adapter es autónomo; pipeline los orquesta |

---

*Documento generado automáticamente el 2026-11-12 desde AST y análisis de imports.*
