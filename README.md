# CCRBroker — routing_data_kit

Kit para **construir y reutilizar datasets de routing LLM** sobre modelos de **pesos abiertos**,
alineado con el marco de contribuciones E1–E8 del proyecto NFDI4Culture.

> 🧭 Objetivo: generar (o reutilizar) datasets benchmark de routing que permitan comparar
> estrategias de routing bajo presupuesto, con métricas estándar (Gain@b, Gap@O, AUDC, AIQ, CRC)
> y soporte para escenarios de drift, agentes multi-turno y adversarios.

## Tabla de contenidos

- [Setup rápido](#setup-rápido)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Fases de investigación](#fases-de-investigación)
- [Uso](#uso)
- [Métricas](#métricas)
- [Desarrollo](#desarrollo)
- [Contribuciones NFDI4Culture](#contribuciones-nfdi4culture)

---

## Setup rápido

```bash
# 1. Clonar
git clone https://github.com/jmschnei/CCRBroker
cd CCRBroker

# 2. Entorno virtual + dependencias
make install          # crea .venv e instala deps base (CPU)
#make install-gpu     # añade vLLM + libs de evaluación (GPU)

# 3. Probar que todo funciona (sin GPU ni red)
make test             # suite completa (test_synthetic + test_agents + test_pipeline)

# 4. Reutilizar datasets públicos (Fase 1)
make phase1

# 5. Generar una fase nueva (ejemplo: Fase 2 ConfigRoute)
make phase2
# ó: PHASE=3 N=500 make run
```

Variables del Makefile: `PHASE` (1–5), `N` (muestras por celda, default 200),
`OUT` (directorio de salida, default `outputs/`), `WARMUP` (warmup de vLLM, default 5).

## Estructura del proyecto

```
CCRBroker/
├── routedata/           # núcleo de datos y métricas
│   ├── schema.py        # Request/Answer/Outcome/ModelScore (pydantic v2)
│   ├── pipeline.py      # carga los 4 datasets → esquema unificado (parquet + manifest)
│   ├── download.py      # descarga de datasets crudos
│   ├── validate.py      # validación estructural de data/raw/ (CLI incluida)
│   ├── adapters/        # RouterBench, SPROUT, LLMRouterBench, R2-Bench, RouterArena
│   ├── metrics.py       # Gain@b, Gap@O, AUDC, AIQ, CRC
│   ├── baselines.py     # best/average/worst oracle, uniform, random, budget-aware
│   ├── conformal.py     # cobertura conformal para Gap@O
│   ├── splits.py        # splits por dominio/idioma/esfuerzo
│   ├── stats.py         # bootstrap + semilla determinística
│   ├── pricing.py       # coste self-host por token (GPU-h)
│   └── agents/          # agentes de routing (6 baselines) + API para routers aprendidos
├── generate/            # generación de datasets (Fases 2–5)
│   ├── backends.py      # vLLM + Mock (sin GPU ni red)
│   ├── prompts.py       # tareas multi-dominio/idioma
│   ├── efforts.py       # low/medium/high (temperature)
│   ├── verifiers.py     # verificadores por tarea
│   ├── judge.py         # judge de LLM (opcional)
│   ├── configroute.py   # Fase 2
│   ├── driftroute.py    # Fase 3
│   ├── agentroute.py    # Fase 4
│   └── advroute.py      # Fase 5
├── catalog/
│   └── dataset_catalog.yaml   # qué reutilizar y qué crear
├── configs/             # parámetros por fase + pool_openweights.yaml
├── examples/
│   ├── routing_agents_demo.py   # routea una query con los 6 agentes (sin datos)
│   ├── inspect_raw_datasets.py  # validación estructural de data/raw/
│   └── load_unified_dataset.py  # estadísticas + integridad del dataset unificado
├── tests/
│   ├── test_synthetic.py        # núcleo: métricas, conformal, generadores (sintético)
│   ├── test_agents.py           # agentes de routing (sin datos)
│   └── test_pipeline.py         # pipeline, adaptadores y validación (usar data/raw si existe)
├── data/                # descargado localmente (ver gitignore): raw/ + processed/unified/
├── regenerate_combined.py  # regenera data/processed/unified/ (3 datasets integrados)
├── estimate_budget.py   # estimación de GPU-horas
├── Makefile             # venv/install/test/lint/format/typecheck/phase1..5
├── pyproject.toml       # packaging + extras [vllm] [eval] [dev]
├── .pre-commit-config.yaml
├── .gitignore
├── LICENSE              # MIT
├── README.md
└── PLAN.md              # plan de investigación completo (Fases 0–5)
```

## Fases de investigación

| Fase | Nombre | Dataset | Eje de contribución |
|------|--------|---------|---------------------|
| 1 | Reutilización | LLMRouterBench, RouterBench, SPROUT (+R2-Bench pendiente, RouterArena por descargar) | E1, E5, E6 |
| 2 | ConfigRoute | (nuevo) configuración × esfuerzo × longitud | E1, E3 |
| 3 | DriftRoute | (nuevo) deriva temporal de preferencias | E2, E5, E6 |
| 4 | AgentRoute | (nuevo) agentes con coste de cambio | E2, E4, E5 |
| 5 | AdvRoute | (nuevo) perturbaciones y ataques | E4, E6, E7 |

Detalles en [`PLAN.md`](PLAN.md).

## Uso

### Agente de routing (6 baselines, sin datos)

```python
from routedata.agents import create_agent, Candidate

agent = create_agent("heuristic")   # random | cost | quality | latency | hybrid | heuristic
candidates = [
    Candidate(name="qwen2.5-7b-instruct", cost_per_1k_tokens=0.05, quality_score=0.62, latency_ms=180),
    Candidate(name="qwen3-30b-a3b",       cost_per_1k_tokens=0.30, quality_score=0.88, latency_ms=900),
]
decision = agent.decide("Explain step by step how transformers work.", candidates)
print(decision.chosen_model, decision.reasoning)
```

Demo ejecutable: `python examples/routing_agents_demo.py`.

### Dataset unificado (Fase 1)

Los datasets crudos viven en `data/raw/` (descarga local, fuera de git). El pipeline
los normaliza a `data/processed/unified/` (Parquet y `manifest.json`). La cifra actual
de filas debe consultarse en ese manifiesto, que es la fuente de verdad; los resúmenes
anteriores pueden estar desactualizados. R2-Bench no se incluye en la regeneración
habitual por su tamaño (ver `PIPELINE_SUMMARY.md`).

```bash
make phase1                             # regenera RouterBench, SPROUT y LLMRouterBench
make phase1-audit                       # verifica pool oficial, cobertura y labels
make phase1-eval                        # seis routers/baselines × cinco semillas, 15 datasets
python -m routedata.pipeline load        # pipeline general; también intenta cargar R2-Bench
make validate                           # validación estructural de data/raw/
make validate-unified                   # recuentos, manifiesto, IDs y métricas de los Parquet
python examples/load_unified_dataset.py  # estadísticas + integridad referencial
python examples/inspect_raw_datasets.py  # resumen de los 4 datasets crudos
```

### Reutilizar datasets públicos (Fase 1)

```python
from routedata.adapters.llmrouterbench import LLMRouterBenchAdapter

queries, configs, responses = LLMRouterBenchAdapter().load()
print(len(queries), len(configs), len(responses))
```

Los adaptadores se importan desde sus módulos concretos; RouterArena es solo para
evaluación y no debe usarse para entrenar o ajustar el router.

La evaluación local usa los 20 modelos del pool performance-only y las 15 tareas
de LLMRouterBench, separa por prompt dentro de cada dataset y excluye queries con
labels `aggregate_fallback`. Los resultados son baselines de este proyecto, no una
reproducción de métodos publicados. Véanse los JSON de auditoría/resultados en
`outputs/phase1/` (artefactos locales ignorados por Git).

La tabla unificada es dispersa: contiene solo respuestas observadas. Algunas etiquetas
de LLMRouterBench son agregadas y están marcadas en `label_source`; respuestas sin score
quedan como `NaN`. Las fuentes históricas no contienen latencias utilizables.

### Generar un dataset nuevo (Fases 2–5)

Los módulos de las fases 2–5 exponen utilidades Python, pero aún no disponen de una
CLI autónoma que configure datos, modelos y backends. Por ello, `make phase2`–`make phase5`
informa explícitamente de ese bloqueo en vez de lanzar comandos inexistentes.
La Fase 1 sí se puede ejecutar con `make phase1` si los datos crudos están disponibles.

### Estimar presupuesto

```bash
python -m estimate_budget --n-queries 200 --n-configs 60
# También está disponible como comando instalado:
routing-budget --n-queries 200 --n-configs 60
# o usar los valores configurables del Makefile:
make budget N_QUERIES=200 N_CFGS=60
```

### Fase 0: medir throughput en una GPU NVIDIA

Las revisiones del pool están fijadas a commits HF. El benchmark requiere Linux con
NVIDIA/CUDA y vLLM; no ejecuta la medición en macOS/Metal. Instala el extra vLLM en
el servidor y aporta el precio real por GPU-hora y su fuente:

```bash
make phase0-measure MODEL=Qwen/Qwen3-0.6B GPU_USD_H=1.80 GPU_PRICE_SOURCE="proveedor y tarifa"
```

Los resultados se añaden a `outputs/phase0/throughput.jsonl` con revisión, hardware,
configuración, tokens/s agregados y costes derivados (incluyendo escenarios ±50 %).
No se deben sustituir los supuestos de hardware/precio del pool hasta verificar esos
datos. El gate de Fase 0 sigue pendiente hasta guardar al menos una medición real.

### Métricas disponibles

```python
from routedata.metrics import gain_at, gap_at, audc, aiq, crc

res = gain_at(ds, strategy, budget_b=0.3)   # Gain@b
res = gap_at(ds, strategy, target_o=0.8)    # Gap@O
res = audc(ds, strategy)                    # Área bajo la curva calidad-vs-coste
res = aiq(ds, strategy)                     # AIQ (área de la curva eficiencia)
res = crc(ds, strategy)                     # Coste relativo de complejidad
```

## Desarrollo

```bash
make lint         # ruff check
make format       # ruff format + fix
make typecheck    # mypy
make test         # pytest
make clean        # borra artefactos locales
make distclean    # clean + .venv/
```

### Pre-commit

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Reglas activas: ruff (lint + format), trailing-whitespace, end-of-file-fixer,
check-yaml, check-json, check-merge-conflict, mixed-line-ending.

## Contribuciones NFDI4Culture

| Eje | Estado |
|-----|--------|
| **E1** — Nuevos datasets (ConfigRoute, DriftRoute) | ✅ generadores implementados |
| **E2** — Dinámica temporal (drift) y de agentes | ✅ Fases 3 y 4 |
| **E3** — Alineación config/modelo | ✅ Fase 2 |
| **E4** — Robustez (perturbaciones/ataques) | ✅ Fase 5 |
| **E5** — Métricas y baselines | ✅ `routedata/metrics.py`, `routedata/baselines.py` |
| **E6** — Splits/evaluación | ✅ `routedata/splits.py`, `routedata/stats.py` |
| **E7** — Datos de seguridad | ✅ `generate/advroute.py` |
| **E8** — Open science (código + catálogo) | ✅ este repo, `catalog/dataset_catalog.yaml` |

---

## Licencia

MIT — ver [`LICENSE`](LICENSE).

**Autores:** Julian Moreno Schneider (DFKI / NFDI4Culture) · 2026
**Repositorio:** https://github.com/jmschnei/CCRBroker
