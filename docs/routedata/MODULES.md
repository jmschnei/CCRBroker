# 📦 Módulo `routedata/` — Core data & metrics

Este módulo contiene todo el núcleo de procesamiento de datos, métricas de evaluación,
agentes de routing y utilidades de infraestructura.

---

## Tabla de contenidos

| Archivo | Descripción | Clases | Funciones |
|---------|-------------|--------|-----------|
| [`__init__.py`](__init__.md) | Entry point del paquete | 0 | 0 |
| [`schema.py`](schema.md) | Pydantic models unificados (Query, Config, Response) | — | 5 |
| [`pipeline.py`](pipeline.md) | Ingestión y unificación de datasets | 1 | 1 |
| [`metrics.py`](metrics.md) | Métricas de evaluación routing | — | 21 |
| [`baselines.py`](baselines.md) | Algoritmos base de routing | 1 | 1 |
| [`conformal.py`](conformal.md) | Control de riesgo conformal (CRC) | — | 6 |
| [`download.py`](download.md) | Descarga y preparación de datasets crudos | — | 5 |
| [`pricing.py`](pricing.md) | Cálculo de costes GPU por token | 1 | 5 |
| [`splits.py`](splits.md) | Estrategias de partición de datos | — | 8 |
| [`stats.py`](stats.md) | Pruebas estadísticas (bootstrap, McNemar...) | — | 6 |
| [`validate.py`](validate.md) | Validación estructural y semántica de datos | 1 | 8 |
| [`drift.py`](drift.md) | Detección y cuantificación de deriva temporal | 4 | 9 |
| [`agent.py`](agent.md) | Utilidades para agentes multi-turno | 2 | 8 |
| [`routerarena.py`](routerarena.md) | Integración con RouterArena benchmark | — | 2 |
| [`routerbench.py`](routerbench.md) | Integración con RouterBench benchmark | — | 1 |
| [`base.py`](base.md) | Clases base y utilidades compartidas | 1 | 2 |

## Submódulos

### `agents/` — Agentes de routing (6 baselines)
- [`__init__.py`](agents/__init__.md) — Factory pattern
- [`base.py`](agents/base.md) — Clases base: RoutingAgent, RoutingDecision, Candidate, RoutingStrategy
- [`baseline.py`](agents/baseline.md) — Implementaciones: random, cost, quality, latency, hybrid, heuristic

### `adapters/` — Adaptadores de datasets externos
- [`__init__.py`](adapters/__init__.md) — Registro de adaptadores
- [`base.py`](adapters/base.md) — Clase base DatasetAdapter
- [`routerbench.py`](adapters/routerbench.md) — Adaptador para RouterBench (pickle)
- [`routerarena.py`](adapters/routerarena.md) — Adaptador para RouterArena (eval_only)
- `sprout.py` — Adaptador SPROUT (columnas: id→query_id, prompt→text)
- `llmrouterbench.py` — Adaptador LLMRouterBench (pool ligero 20 modelos)
- `r2bench.py` — Adaptador R2-Bench (30k queries × 15 LLM × 16 budgets)

---

## Relaciones entre submódulos

```
schema.py ────────► pipeline.py ────────► validate.py
     │                   │                      │
     ▼                   ▼                      ▼
metrics.py ◄──────── baselines.py          splits.py
     │                    │                    │
     ▼                    ▼                    ▼
conformal.py         agent.py            stats.py
     │                   │
     ▼                   ▼
pricing.py          drift.py (Fase 3 core)

agents/base.py ───► agents/baseline.py (6 agentes concretos)
adapters/base.py ──► adapters/routerbench/sprout/etc. (4 adaptadores concretos)
```

---

## Funciones clave por categoría

### 📊 Métricas (metrics.py)
| Función | Descripción | Uso principal |
|---------|-------------|---------------|
| `gain_at_b(q, c, b)` | Gain@b: mejora relativa con budget b | E1, RQ1 |
| `gap_at_o(q, c, oracle)` | Brecha al oracle global | E2, RQ2 |
| `aiq(approx=True)` | Accuracy-weighted Quality | E4, RQ4 |
| `audc(response_df)` | Área bajo curva de decisión | E3, RQ3 |
| `ece_brier(preds, actuals)` | Calibración de probabilidades | E5 |

### ⚙️ Estadísticas (stats.py)
| Función | Descripción | Uso principal |
|---------|-------------|---------------|
| `bootstrap_ci(data, n_boot=1000)` | Intervalos de confianza bootstrap | Todas |
| `mcnemar_exact(a, b)` | Test exacto de McNemar | Fase 3 drift |
| `holm_correction(p_values)` | Corrección de Holm (FWER) | Comparaciones múltiples |
| `bh_correction(p_values)` | Benjamini-Hochberg (FDR) | Análisis exploratorio |

### 🔄 Split strategies (splits.py)
| Función | Descripción | Uso principal |
|---------|-------------|---------------|
| `split_iid(df, test_frac, seed)` | Partición IID | Base line |
| `split_cluster(df, k_clusters, seed)` | Por clusters semánticos | RQ8 multilingüe |
| `split_temporal(df, cutoff_date)` | Temporal (train antes de cutoff) | DriftRoute |

### 🎯 Agentes (agents/baseline.py)
| Agente | Estrategia | Descripción |
|--------|------------|-------------|
| `random` | Aleatorio | Baseline mínimo |
| `cost` | Menor coste | Siempre el más barato |
| `quality` | Mayor calidad | Siempre el de mejor score |
| `latency` | Menor latencia | Siempre el más rápido |
| `hybrid` | Coste + calidad | Combina ambos factores |
| `heuristic` | Reglas heurísticas | Dominio, idioma, complejidad |

### 🌡️ Drift detection (drift.py)
| Función | Descripción | Uso principal |
|---------|-------------|---------------|
| `freeze_subset(queries, n_queries)` | Congela subconjunto estratificado | init DriftRoute |
| `split_train_test(subset, frac)` | Partición fija train/test (cluster) | init DriftRoute |
| `drift_report(resp_a, resp_b)` | Δ precisión y Δ tokens entre snaps | report DriftRoute |
| `prequential(snaps, train_ids, test_ids)` | Evaluación prequential de routers | eval E4 |

### 💰 Pricing (pricing.py)
| Función | Descripción | Uso principal |
|---------|-------------|---------------|
| `derive_selfhost_prices()` | Deriva precios desde GPU-h y tok/s | Fase 0 |
| `scale_prices(prices, factor)` | Escenario ±50% de coste | sensitivity analysis |
| `reprice_configs(cfgs, throughputs)` | Repricing de configs completas | cada snapshot Phase3 |

---

*Documentación generada automáticamente el 2026-11-12 desde AST del código fuente.*
