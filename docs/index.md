# 📚 routedata_kit — Documentación completa

Kit para **construir y reutilizar datasets de routing LLM** sobre modelos de **pesos abiertos**,
alineado con el marco de contribuciones E1–E8 del proyecto NFDI4Culture.

> 🧭 **Objetivo:** generar (o reutilizar) datasets benchmark de routing que permitan comparar
> estrategias de routing bajo presupuesto, con métricas estándar (Gain@b, Gap@O, AUDC, AIQ, CRC)
> y soporte para escenarios de drift, agentes multi-turno y adversarios.

---

## 📖 Tabla de contenidos

- [Arquitectura del sistema](architecture.md)
- [Módulo `routedata/`](routedata/MODULES.md) — Core data & metrics
- [Módulo `generate/`](generate/MODULES.md) — Dataset generation phases
- [Scripts de ejecución](scripts/)
- [Ejemplos de uso](examples/)

---

## 🏗️ Arquitectura general

```
routedata_kit/
├── routedata/                 # Núcleo: datos, métricas, agentes
│   ├── schema.py              # Pydantic models (Query, Config, Response)
│   ├── pipeline.py            # Data ingestion pipeline
│   ├── metrics.py             # Gain@b, Gap@O, AIQ, AUDC...
│   ├── baselines.py           # Baseline routing algorithms
│   ├── conformal.py           # Conformal risk control (CRC)
│   ├── download.py            # Dataset downloading utilities
│   ├── pricing.py             # GPU pricing calculations
│   ├── splits.py              # Split strategies
│   ├── stats.py               # Statistical tests (bootstrap, McNemar...)
│   ├── validate.py            # Data validation framework
│   ├── drift.py               # Drift detection temporal
│   ├── agent.py               # Agent utilities
│   ├── routerarena.py         # RouterArena integration
│   ├── routerbench.py         # RouterBench integration
│   ├── base.py                # Base classes/utilities
│   ├── agents/                # Routing agents (6 baselines)
│   │   ├── base.py            # Base classes & enums
│   │   └── baseline.py        # random, cost, quality, latency, hybrid, heuristic
│   └── adapters/              # Dataset adapters
│       ├── base.py            # Adapter base class
│       ├── routerbench.py     # RouterBench adapter
│       ├── routerarena.py     # RouterArena adapter
│       ├── sprout.py          # SPROUT adapter (in source, documented here)
│       ├── llmrouterbench.py  # LLMRouterBench adapter (in source)
│       └── r2bench.py         # R2-Bench adapter (in source)
├── generate/                  # Dataset generation framework (Fases 2–5)
│   ├── __init__.py            # Module entry point
│   ├── advroute.py            # Adversarial routing (Phase 5)
│   ├── agentroute.py          # Agent-based routing (Phase 4)
│   ├── backends.py            # vLLM + Mock backend implementations
│   ├── build_queries.py       # Query generation utilities
│   ├── configroute.py         # Configuration route generation (Phase 2 core)
│   ├── driftroute.py          # Drift route generation (Phase 3 core)
│   ├── efforts.py             # Effort/budget forcing logic
│   ├── judge.py               # LLM judge utilities
│   ├── multilingual.py        # Multilingual query support
│   ├── phase2.py              # Phase 2 specific code
│   ├── prompts.py             # Prompt templates for all domains
│   ├── run_phase2.py          # Phase 2 CLI orchestrator
│   ├── run_phase3.py          # Phase 3 CLI orchestrator
│   ├── run_phase4.py          # Phase 4 CLI orchestrator
│   ├── verifiers.py           # Task verifiers per domain
│   └── envs/                  # Agent environment support
│       ├── __init__.py        # Environment registry
│       ├── synthetic.py       # Synthetic test environment
│       └── adapters_template.py  # Real adapter templates (τ², ScienceWorld, etc.)
├── scripts/                   # Standalone scripts
│   ├── benchmark_vllm.py      # GPU throughput measurement
│   ├── run_phase1.py          # Phase 1 execution script
│   └── run_phase1_multidataset.py  # Multi-dataset phase 1 runner
├── examples/                  # Usage examples
│   ├── inspect_raw_datasets.py    # Data inspection
│   ├── load_unified_dataset.py    # Unified dataset loading
│   └── routing_agents_demo.py     # Agent demonstration
├── configs/                   # Configuration files
├── catalog/                   # Dataset catalog
├── data/                      # Raw & processed data (gitignored)
├── tests/                     # Test suite
├── outputs/                   # Experiment artifacts
└── PLAN.md                    # Research plan (phases 0–5)
```

---

## 🚀 Quick Start

```bash
# 1. Instalación con GPU support
make install && pip install -e '.[vllm]'

# 2. Ejecutar tests base (sin GPU)
make test

# 3. Fase 0: Medición de throughput GPU
make phase0-me MODEL="Qwen/Qwen3-0.6B" GPU_USD_H="1.8" GPU_PRICE_SOURCE="aws-a100"

# 4. Fase 1: Dataset unificado + validación
make phase1 && make validate-unified && make phase1-eval

# 5. Fase 2 piloto (ConfigRoute)
python -m generate.run_phase2 build-queries --cfg configs/configroute_pilot.yaml --out data/queries
python -m generate.run_phase2 pilot --pool configs/pool_openweights.yaml --cfg configs/configroute_pilot.yaml --queries data/queries --out data/pilot --backend vllm
python -m generate.run_phase2 judge --run-dir data/pilot --pool configs/pool_openweights.yaml --backend vllm
python -m generate.run_phase2 report --run-dir data/pilot --pool configs/pool_openweights.yaml --cfg configs/configroute_pilot.yaml --n-full 8000

# 6. Fase 3 (DriftRoute)
python -m generate.run_phase3 init --cfg configs/driftroute.yaml --drift-dir data/drift --queries data/processed/unified/queries_combined.parquet
python -m generate.run_phase3 snapshot --drift-dir data/drift --snapshot s00 --pool configs/pool_openweights.yaml --date 2026-11-15 --backend vllm

# 7. Fase 4 (AgentRoute) simulación
python -m generate.run_phase4 simulate --n-tasks 60
```

---

## 📊 Métricas implementadas

| Métrica | Descripción | Fases |
|---------|-------------|-------|
| **Gain@b** | Aumento relativo de precisión con budget b | E1, RQ1 |
| **Gap@O** | Brecha entre router y oracle | E2, RQ2 |
| **AIQ** | Accuracy-weighted Quality (aproximada) | E4, RQ4 |
| **AUDC** | Area Under Decision Curve | E3, RQ3 |
| **QNC** | Quality-Neutrality Coverage | E3, RQ8 |
| **CRC** | Conformal Risk Control | E7, RQ7 |
| **Pareto/Hypervolume** | Pareto front analysis | E1, E6 |
| **ECE/Brier** | Calibration metrics | E5 |

---

## 🧭 Fases de investigación

| Fase | Nombre | Dataset | Contribución | Estado código |
|------|--------|---------|--------------|---------------|
| 0 | Infraestructura | — | Throughput medido, precios derivados | ⚠️ Requiere GPU real |
| 1 | Reutilización | LLMRouterBench, RouterBench, SPROUT | E1, E5, E6 | ✅ Implementado |
| 2 | ConfigRoute | (nuevo) config × effort × length | E1, E3, E7, E8 | ⚠️ Pilot ready |
| 3 | DriftRoute | (nuevo) temporal drift | E2, E4, E6 | ✅ Framework listo |
| 4 | AgentRoute | (nuevo) agent cost of switching | E2, E4, E5 | ⚠️ Core implemented |
| 5 | AdvRoute | (nuevo) perturbations & attacks | E4, E6, E7 | ⚠️ Structure only |

---

## 📁 Navegación por módulos

### [`routedata/`](routedata/MODULES.md) — Core data infrastructure
- **Schema:** Pydantic models para Queries, Configs, Responses
- **Pipeline:** Ingestión y unificación de datasets (3+1 fuentes)
- **Metrics:** Gain@b, Gap@O, AIQ, AUDC, CRC, Pareto, ECE...
- **Baselines:** Algoritmos base de routing
- **Conformal:** Control de riesgo conformal para garantía estadística
- **Download:** Utilidades de descarga de datasets crudos
- **Pricing:** Cálculo de costes GPU por token
- **Splits:** Estrategias de partición (iid, cluster-based, temporal...)
- **Stats:** Bootstrap, McNemar exacto, Holm, Benjamini-Hochberg
- **Validate:** Validación estructural y semántica de datos
- **Drift:** Detección y cuantificación de deriva temporal
- **Agent:** Utilidades para agentes multi-turno
- **Agents:** 6 agentes de routing baseline (random, cost, quality...)
- **Adapters:** Adaptadores para RouterBench, SPROUT, LLMRouterBench...

### [`generate/`](generate/MODULES.md) — Dataset generation framework
- **Backends:** Implementaciones vLLM y Mock para inferencia
- **ConfigRoute:** Generación de configs (model × effort × length)
- **DriftRoute:** Serie temporal de snapshots mensuales
- **AgentRoute:** Enrutado multi-turno con coste de cambio
- **AdvRoute:** Perturbaciones y ataques adversarios
- **BuildQueries:** Construcción de queries desde fuentes externas
- **Efforts:** Lógica de forcing por presupuesto de tokens
- **Judge:** LLM judge para tareas abiertas
- **Multilingual:** Soporte ES/DE/FR para queries
- **Phase2:** Código específico de la Fase 2
- **Prompts:** Plantillas de prompt multi-dominio
- **Verifiers:** Verificadores programáticos por tipo de tarea
- **Envs:** Entornos sintéticos y plantillas de adaptadores reales

### [Scripts](scripts/) — Standalone execution tools
- `benchmark_vllm.py` — Medición de throughput GPU individual por modelo
- `run_phase1.py` — Ejecución directa de baselines Fase 1
- `run_phase1_multidataset.py` — Auditoría + evaluación multi-dataset

### [Examples](examples/) — Usage demos
- `inspect_raw_datasets.py` — Inspección y validación de data/raw/
- `load_unified_dataset.py` — Carga del dataset unificado con estadísticas
- `routing_agents_demo.py` — Demostración de los 6 agentes routing

---

## 🔧 Referencia rápida de interfaces clave

### Schema (routedata/schema.py)
```python
class Query(BaseModel):
    query_id: str
    text: str
    answer: str
    domain: Optional[str] = None
    category: Optional[str] = None
    source_dataset: str

class Config(BaseModel):
    config_id: str
    model: str
    revision: str
    effort: str
    budget_tokens: Optional[int] = None
    price_in: float
    price_out: float

class Response(BaseModel):
    query_id: str
    config_id: str
    correct: float
    cost_usd: float
```

### RouterAgent API (routedata/agents/baseline.py)
```python
from routedata.agents import create_agent, Candidate

agent = create_agent("heuristic")
candidates = [
    Candidate(name="qwen2.5-7b", cost=0.05, quality=0.62),
    Candidate(name="qwen3-30b", cost=0.30, quality=0.88),
]
decision = agent.decide("Explain transformers.", candidates)
```

### Phase2 CLI (generate/run_phase2.py)
```bash
python -m generate.run_phase2 build-queries --cfg ...
python -m generate.run_phase2 pilot --pool ... --queries ...
python -m generate.run_phase2 judge --run-dir ...
python -m generate.run_phase2 report --run-dir ... --n-full 8000
```

### Phase3 CLI (generate/run_phase3.py)
```bash
python -m generate.run_phase3 init --cfg ... --drift-dir data/drift
python -m generate.run_phase3 snapshot --drift-dir ... --snapshot s00
python -m generate.run_phase3 check --drift-dir data/drift
python -m generate.run_phase3 eval --drift-dir data/drift
```

### Phase4 CLI (generate/run_phase4.py)
```bash
python -m generate.run_phase4 init --data-dir data/agent --env synthetic
python -m generate.run_phase4 make-prices --data-dir ... --configs ...
python -m generate.run_phase4 collect --data-dir ... --endpoints ...
python -m generate.run_phase4 stepdata --data-dir ... --mode local|rollout
python -m generate.run_phase4 evaluate --data-dir ... --lams "10,30,100"
python -m generate.run_phase4 analyze --data-dir data/agent
```

---

## 📊 Dependencias entre fases

```
Fase 0 (GPU → precios)
    ↓
Fase 1 (dataset unificado + auditoría)
    ↓
Fase 2 piloto ────────→ Fase 3 (snapshots mensuales)
    ↓                          ↓
Fase 2 escala                E4 evaluation
    ↓
Fase 4 (AgentRoute) ────────→ Fase 5 (AdvRoute) [independiente]
```

---

## 📝 Licencia

MIT © Julian Moreno Schneider (DFKI / NFDI4Culture)

---

*Generada automáticamente el 2026-11-12 desde AST del código fuente.*
