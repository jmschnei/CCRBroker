# 🏗️ Arquitectura del sistema routedata_kit

Este documento describe la arquitectura de alto nivel, patrones de diseño y flujo de datos
del kit `routing_data_kit` (CCRBroker) para benchmarks de routing LLM.

---

## 1. Visión general

```
┌───────────────────────────────────────────────────────────────┐
│                    routedata_kit                              │
│                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │  Phase 0     │   │  Phase 1     │   │  Phases 2–5     │ │
│  │ (Infra)      │──▶│ (Reutilización│   │ (Generación new) │ │
│  └──────────────┘   └──────────────┘   └──────────────────┘ │
│       │                    │                    │            │
│       ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   routedata/ (Core)                     │ │
│  │  Schema → Pipeline → Metrics → Baselines → Conformal   │ │
│  └─────────────────────────────────────────────────────────┘ │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 generate/ (Phases)                      │ │
│  │  ConfigRoute · DriftRoute · AgentRoute · AdvRoute      │ │
│  └─────────────────────────────────────────────────────────┘ │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Output Artifacts                           │ │
│  │  unified/ · pilot/ · drift/ · agent/ · adv/           │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. Capas del sistema

### Capa 1: Schema & Types (routedata/schema.py)
- **Responsabilidad:** Definir el modelo unificado de datos
- **Clases principales:** `Query`, `Config`, `Response`, `Outcome`
- **Patrón:** Pydantic models con validación estricta
- **Flujo:** Entrada → Schemas → Pipeline → Unificación

### Capa 2: Ingestión & Validación (routedata/pipeline.py, validate.py)
- **Responsabilidad:** Cargar datasets crudos y unificarlos
- **Adaptadores:** RouterBench, SPROUT, LLMRouterBench, R2-Bench
- **Validación:** Esquemática (PKs, FKs) + semántica (rango de valores)

### Capa 3: Métricas & Análisis (routedata/metrics.py, stats.py)
- **Responsabilidad:** Medir calidad, coste y eficiencia del routing
- **Métricas:** Gain@b, Gap@O, AIQ, AUDC, QNC, CRC, Pareto...
- **Estadísticas:** Bootstrap, McNemar exacto, Holm, Benjamini-Hochberg

### Capa 4: Agents & Baselines (routedata/agents/baseline.py)
- **Responsabilidad:** Proveer estrategias de routing de referencia
- **Agentes:** random, cost, quality, latency, hybrid, heuristic
- **Patrón:** Estrategia con factory (`create_agent()`)

### Capa 5: Generación por Fases (generate/)
- **Fase 2 ConfigRoute:** model × effort × budget grid
- **Fase 3 DriftRoute:** series temporal snapshots mensuales
- **Fase 4 AgentRoute:** enrutado multi-turno con coste de cambio
- **Fase 5 AdvRoute:** perturbaciones y ataques adversarios

### Capa 6: Backends & Infra (generate/backends.py, scripts/benchmark_vllm.py)
- **Responsabilidad:** Abstracción del motor de inferencia
- **Implementaciones:** VLLMBackend (real), MockBackend (test/simulación)
- **Precios:** Derivados de throughput GPU medido

---

## 3. Flujo de datos principal

```
Dataset crudo (RouterBench/SPROUT/LLMRouterBench)
    │
    ▼
routedata/download.py (descarga desde HF, APIs públicas)
    │
    ▼
routedata/adapters/* (normalización de esquemas)
    │
    ▼
routedata/pipeline.py (unificación → Parquet)
    │
    ├── queries_combined.parquet
    ├── configs_combined.parquet
    └── responses_combined.parquet
    │
    ▼
Validación: routedata/validate.py + make validate-unified
    │
    ▼
Métricas: metrics.py (Gain@b, Gap@O, etc.) + stats.py (bootstrap)
    │
    ├── Baselines locales
    ├── Evaluación en RouterArena sub_10
    └── Auditoría oficial protocol_audit.json
    │
    ▼
Generación de datasets nuevos (Fases 2–5):
    ├── Phase2 → config × effort × budget grid
    ├── Phase3 → series temporal con eventos
    ├── Phase4 → agent episodes con step utilities
    └── Phase5 → adversarial perturbations
    │
    ▼
Reportes y artefactos:
    ├── outputs/phase1/*
    ├── data/pilot/REPORT.md
    ├── data/drift/E4_REPORT.md
    └── data/agent/AGENT_REPORT.md
```

---

## 4. Patrones de diseño clave

### Patrón Strategy (Agentes)
```python
# routedata/agents/baseline.py
class RoutingStrategy(Enum):
    RANDOM = "random"
    COST = "cost"
    QUALITY = "quality"
    LATENCY = "latency"
    HYBRID = "hybrid"
    HEURISTIC = "heuristic"

# Factory pattern
def create_agent(strategy: str) -> RoutingAgent:
    ...
```

### Patrón Adapter (Datasets)
```python
# routedata/adapters/base.py → routerbench.py, sprout.py, etc.
class DatasetAdapter(ABC):
    @abstractmethod
    def load_raw(self, path: Path) -> DataFrame:
        pass

    @abstractmethod
    def normalize_schema(self, df: DataFrame) -> DataFrame:
        pass

# Implementaciones concretas
class RouterBenchAdapter(DatasetAdapter): ...
class SPROUTAdapter(DatasetAdapter): ...
```

### Patrón Template Method (Fases)
```python
# generate/run_phase*.py - misma interfaz CLI para todas las fases
subparsers.add_parser('init')    # inicialización
subparsers.add_parser('snapshot'/'collect')  # generación principal
subparsers.add_parser('check')   # validación intermedia
subparsers.add_parser('report'/'analyze')  # reporte final
```

### Patrón Builder (Queries & Configs)
```python
# generate/configroute.py
sample_queries(sources, quotas) → DataFrame  # builder de queries
build_config_grid(models, budgets, efforts) → DataFrame  # builder de configs
run(queries, configs, backend) → List[Response]  # ejecución
```

---

## 5. Relaciones entre módulos

### Dependencias directas (solo core)
```
routedata/__init__ → schema + pipeline + validate
    ├── metrics.py ← baselines, conformal, splits
    ├── stats.py   ← (estadísticas puras)
    ├── splits.py  ← (lógica de partición)
    ├── conformal.py ← pricing
    ├── pricing.py ← (cálculos GPU)
    ├── download.py ← (utilidades externas)
    └── validate.py → schema, pipeline, baselines
```

### Dependencias phase→core
```
generate/phase2.py ← configroute.py + backends + schema + stats
generate/run_phase2.py ← phase2.py + build_queries.py + judge.py + verifiers.py
generate/driftroute.py ← configroute.py + drift.py + pricing.py + stats
generate/run_phase3.py ← driftroute.py + configroute.py + schema
generate/agentroute.py ← schema + agent.py (utilidades) + backends
generate/run_phase4.py ← agentroute.py + envs/* + schemas
generate/multilingual.py ← utils (traducción de queries)
generate/judge.py ← backends + schema
generate/verifiers.py ← (lógica por dominio)
generate/prompts.py ← templates (sin dependencias externas)
```

---

## 6. Estructura de datos unificada

### Esquema Pydantic (routedata/schema.py)

```python
# queries_combined.parquet
Query:
    query_id: str          # ID único
    text: str              # Texto de la pregunta
    answer: str            # Respuesta gold
    domain: Optional[str]  # Dominio (math, code, science...)
    category: Optional[str] # Categoría específica
    language: Optional[str] # Idioma (en, es, de...)
    source_dataset: str    # Fuente (routerbench, sprout, llmrouterbench)

# configs_combined.parquet
Config:
    config_id: str         # ID único de config
    model: str             # Nombre del modelo
    revision: str          # SHA HF exacto
    effort: str            # nothink / think_low / ...
    budget_tokens: Optional[int]  # Preset de longitud (null=default)
    price_in: float        # Coste USD/MTok input
    price_out: float       # Coste USD/MTok output
    price_source: str      # Fuente del precio
    price_date: str        # Fecha de consulta

# responses_combined.parquet
Response:
    query_id: str          # FK → queries.query_id
    config_id: str         # FK → configs.config_id
    run_id: str            # Identificador de ejecución
    snapshot_id: Optional[str]  # Para drift series
    correct: Optional[float]   # [0,1] label de calidad
    cost_usd: Optional[float]  # Coste final de la respuesta
    in_tokens: Optional[int]   # Tokens input
    out_tokens: Optional[int]  # Tokens output
    latency_s: Optional[float] # Latencia (generalmente NaN en histórico)
```

---

## 7. Diagrama de clases simplificado

### Core classes (routedata/)

```
┌─────────────────────┐     ┌──────────────────────┐
│   DatasetAdapter    │     │   RoutingAgent       │
│  (ABC, adapters/   │◄────│  (ABC, agents/base)  │
│                    │     │                      │
│ +load_raw()        │     │ +decide(question,    │
│ +normalize_schema()│     │   candidates)         │
└─────────────────────┘     └──────────┬───────────┘
                                       │
                        ┌──────────────┼──────────────┐
                        │              │              │
                   ┌─────────┐   ┌─────────┐  ┌─────────┐
                   │RandomAgent│ │CostAgent│  │...       │
                   └─────────┘   └─────────┘  └─────────┘

┌─────────────────────┐     ┌──────────────────────┐
│      Query          │     │       Response       │
│ (pydantic.BaseModel)│     │ (pydantic.BaseModel) │
│                     │     │                      │
│ +query_id: str      │     │ +query_id: str       │
│ +text: str          │◄────┤ +config_id: str      │
│ +answer: str        │     │ +correct: float      │
│ ...                 │     │ +cost_usd: float     │
└─────────────────────┘     └──────────────────────┘

┌─────────────────────┐     ┌──────────────────────┐
│      Config         │     │       Metrics        │
│ (pydantic.BaseModel)│     │  (static functions)  │
│                     │     │                      │
│ +config_id: str     │     │ +gain_at_b()         │
│ +model: str         │◄────┤ +gap_at_o()          │
│ +effort: str        │     │ +aiq()               │
│ ...                 │     │ +audc()              │
└─────────────────────┘     └──────────────────────┘
```

### Generation framework (generate/)

```
┌──────────────────────────────────────────────────────┐
│                   Backend base                       │
│                  (ABC, backends.py)                  │
│                                                      │
│  +generate(reqs: List[GenRequest]) -> List[Result]  │
└────────────────┬─────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
   ┌──────────┐     ┌──────────┐
   │VLLMBackend│    │MockBackend│
   └──────────┘     └──────────┘

┌──────────────────────────────────────────────────────┐
│                 Phase orchestrator                    │
│               (run_phase2/3/4.py)                   │
│                                                      │
│  CLI subcommands: init, snapshot/collect,             │
│                  check, report/analyze, evaluate      │
└──────────────────────────────────────────────────────┘
```

---

## 8. Puntos de entrada principales (entry points)

| Archivo | Función principal | Uso |
|---------|------------------|-----|
| `routedata/schema.py` | Esquemas Pydantic | Importar y validar datos |
| `routedata/pipeline.py` | `load_dataset()` | Cargar datasets crudos |
| `routedata/metrics.py` | `gain_at_b()`, `gap_at_o()` | Evaluar routers |
| `routedata/agents/baseline.py` | `create_agent()` | Instanciar agentes |
| `scripts/run_phase1_multidataset.py` | `main()` | Ejecutar Fase 1 completa |
| `scripts/benchmark_vllm.py` | `main()` | Medir throughput GPU |
| `generate/run_phase2.py` | CLI subparsers | Orquestar Fase 2 |
| `generate/run_phase3.py` | CLI subparsers | Orquestar Fase 3 |
| `generate/run_phase4.py` | CLI subparsers | Orquestar Fase 4 |
| `regenerate_combined.py` | `main()` | Regenerar dataset unificado |

---

*Documento generado a partir del código fuente el 2026-11-12.*
