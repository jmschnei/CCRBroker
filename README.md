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
make test             # 7 pruebas OK

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
│   ├── adapters/        # LLMRouterBench, RouterBench, SPROUT, RouterArena
│   ├── metrics.py       # Gain@b, Gap@O, AUDC, AIQ, CRC
│   ├── baselines.py     # best/average/worst oracle, uniform, random, budget-aware
│   ├── conformal.py     # cobertura conformal para Gap@O
│   ├── splits.py        # splits por dominio/idioma/esfuerzo
│   ├── stats.py         # bootstrap + semilla determinística
│   └── pricing.py       # coste self-host por token (GPU-h)
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
├── tests/
│   └── test_synthetic.py      # 7 pruebas sin GPU/red
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
| 1 | Reutilización | LLMRouterBench, RouterBench, SPROUT, RouterArena | E1, E5, E6 |
| 2 | ConfigRoute | (nuevo) configuración × esfuerzo × longitud | E1, E3 |
| 3 | DriftRoute | (nuevo) deriva temporal de preferencias | E2, E5, E6 |
| 4 | AgentRoute | (nuevo) agentes con coste de cambio | E2, E4, E5 |
| 5 | AdvRoute | (nuevo) perturbaciones y ataques | E4, E6, E7 |

Detalles en [`PLAN.md`](PLAN.md).

## Uso

### Reutilizar datasets públicos (Fase 1)

```python
from routedata.adapters import LLMRouterBench, RouterBench, SPROUT, RouterArena
from routedata.baselines import oracle_best, uniform, budget_aware
from routedata.metrics import gain_at, gap_at, audc, aiq, crc

ds = LLMRouterBench().load()          # o RouterBench(), SPROUT(), RouterArena()
res = gain_at(ds, oracle_best)
print(res.summary)                    # Gain@b
```

### Generar un dataset nuevo (Fases 2–5)

```bash
# Fase 2: ConfigRoute (nuevos prompts × modelos × esfuerzo)
python -m generate.configroute --n 200 --out outputs/phase2

# Fase 3: DriftRoute (preferencias que cambian en el tiempo)
python -m generate.driftroute  --n 200 --out outputs/phase3

# Fase 4: AgentRoute (agentes con coste de cambio)
python -m generate.agentroute  --n 200 --out outputs/phase4

# Fase 5: AdvRoute (perturbaciones + ataques)
python -m generate.advroute    --n 200 --out outputs/phase5
```

Cada generador escribe JSONL por modelo + metadata + presupuesto estimado.

### Estimar presupuesto

```bash
python estimate_budget.py --n 200 --warmup 5
# o: make budget
```

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
