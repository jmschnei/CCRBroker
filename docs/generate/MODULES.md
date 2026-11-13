# 📦 Módulo `generate/` — Dataset generation framework

Este módulo implementa el framework de generación de datasets para las Fases 2–5, 
incluyendo orquestadores CLI para cada fase.

---

## Tabla de contenidos

| Archivo | Descripción | Clases | Funciones | Fase |
|---------|-------------|--------|-----------|------|
| [`__init__.py`](__init__.md) | Entry point del paquete | 0 | 0 | — |
| [`advroute.py`](advroute.md) | Perturbaciones y ataques adversarios | — | 8 | 5 |
| [`agentroute.py`](agentroute.md) | Enrutado multi-turno con coste de cambio | 6 | 15 | 4 |
| [`backends.py`](backends.md) | Abstracciones vLLM + Mock | 6 | 0 | Todas |
| [`build_queries.py`](build_queries.md) | Construcción de queries desde fuentes externas | — | 5 | 2 |
| [`configroute.py`](configroute.md) | Generación config × effort × budget (core Fase2) | — | 5 | 2 |
| [`driftroute.py`](driftroute.md) | Series temporal snapshots mensuales (core Fase3) | 1 | 10 | 3 |
| [`efforts.py`](efforts.md) | Lógica de forcing por presupuesto de tokens | 1 | 1 | 2 |
| [`judge.py`](judge.md) | LLM judge para tareas abiertas | — | 4 | 2 |
| [`multilingual.py`](multilingual.md) | Soporte ES/DE/FR para queries | — | 5 | 2 |
| [`phase2.py`](phase2.md) | Código específico de la Fase 2 | — | 8 | 2 |
| [`prompts.py`](prompts.md) | Plantillas de prompt multi-dominio | — | 3 | 2 |
| [`run_phase2.py`](run_phase2.md) | Orquestador CLI Fase 2 | 1 | 3 | 2 |
| [`run_phase3.py`](run_phase3.md) | Orquestador CLI Fase 3 | — | 12 | 3 |
| [`run_phase4.py`](run_phase4.md) | Orquestador CLI Fase 4 | — | 17 | 4 |
| [`verifiers.py`](verifiers.md) | Verificadores programáticos por tipo de tarea | — | 8 | 2 |
| `envs/__init__.py` | Registro de entornos | — | 1 | 4 |
| `envs/synthetic.py` | Entorno sintético para pruebas | 1 | 0 | 4 |
| `envs/adapters_template.py` | Plantillas adaptadores reales (τ², ScienceWorld) | 4 | 0 | 4 |

---

## Resumen por fase

### Fase 2: ConfigRoute (configroute.py + run_phase2.py)
```
build_queries() ──► sample_queries(sources, quotas) → DataFrame queries
                  ──► build_config_grid(models, budgets, efforts) → DataFrame configs
run(queries, configs, backend) → List[Response]

Orquestador CLI:
  build-queries   → genera data/queries/
  pilot           → ejecuta ~1000 queries × configs (data/pilot/)
  judge           → etiqueta tareas abiertas con juez LLM
  audit-export    → exporta hoja ciega para auditoría humana
  report          → GENERA REPORT.md con puertas de paso
```

### Fase 3: DriftRoute (driftroute.py + run_phase3.py)
```
freeze_subset(queries, n_queries) → queries congeladas por cluster
split_train_test(subset, frac) → train_ids, test_ids (fijos)
drift_report(resp_a, resp_b) → Δ precisión y Δ tokens

Orquestador CLI:
  init        → congela subset + split train/test (UNICA VEZ)
  new-pool    → genera pool siguiente snapshot con eventos registrados
  snapshot    → ejecuda inferencias + manifiesto + reprecio por throughput
  check       → verifica comparabilidad entre snapshots consecutivos
  report      → deriva por config entre snaps con McNemar+Holm
  eval        → E4: evaluación prequential de routers (static, refit, profile)
  simulate    → ensayo en seco con datos sintéticos + MockBackend
```

### Fase 4: AgentRoute (agentroute.py + run_phase4.py)
```
run_episode(fac(), backend, policy) → Episode(contrafactuales)
collect_step_utilities() → step utilities por estado y modelo
TaskRouter.fit(X, S, C) → router por tarea (kNN/Ridge)
StepRouter.fit(stepdata) → router por paso con coste de cambio

Orquestador CLI:
  init        → tareas + semillas + partición train/test por tarea
  make-prices → precios desde Fase2/3 configs + throughput medido
  collect     → episodios fijos (todas tareas) + mixtos (train)
  stepdata    → etiquetas contrafactuales (mode=local o rollout)
  evaluate    → entrena routers + selecciona λ en validación + eval en test
  analyze     → GENERA AGENT_REPORT.md con H5 results + headroom
  simulate    → simulación completa con entorno sintético + mock
```

### Fase 5: AdvRoute (advroute.py)
Estructura básica para perturbaciones beningnas y ataques black-box.
- **Perturbaciones:** typos, whitespace, pad, option_shuffle, paraphrase, translate_es_de_fr
- **Ataques:** gadget_black_box (length × iters iterations)
- **Objetivos:** inflate_cost, degrade_quality

---

## Dependencias cruzadas con routedata/

```
generate/configroute.py ← routedata/schema + stats + splits
generate/driftroute.py  ← routedata/schema + pricing + drift + stats
generate/agentroute.py  ← routedata/schema + agent (utilities)
generate/backends.py    ← abstracción de inferencia (vLLM o Mock)

generate/run_phase2.py  ← configroute + build_queries + phase2 + judge + verifiers
generate/run_phase3.py  ← driftroute + configroute + schema + pricing
generate/run_phase4.py  ← agentroute + envs/* + schema + stats

Todos los orquestadores ← run_time → backend inference (vLLM real o Mock)
```

---

## Diagrama de clases del framework generation

```
┌──────────────────────┐     ┌──────────────────────┐
│   GenRequest         │     │   GenResult           │
│ (backends.py)        │     │ (backends.py)         │
│                      │     │                       │
│ +messages: List[Dict]│     │ +response_text: str    │
│ +max_tokens: int     │◄────┤ +in_tokens: int       │
│ +temperature: float  │     │ +out_tokens: int      │
│ +seed: int           │     │ +latency_ms: float    │
└──────────────────────┘     └──────────────────────┘

┌────────────────────────────────────────────────────────┐
│                  Backend (ABC)                         │
│                 backends.py                            │
│                                                        │
│  +generate(reqs: List[GenRequest]) → List[GenResult] │
└────────────────────────┬───────────────────────────────┘
                        │
              ┌─────────┴──────────┐
              │                    │
         ┌──────────┐      ┌──────────┐
         │VLLMBackend│     │MockBackend│
         └──────────┘      └──────────┘

┌──────────────────────┐     ┌──────────────────────┐
│   SyntheticAgentEnv  │     │  Env (ABC envs/)     │
│  (envs/synthetic.py) │     │  (adapters_template) │
│                      │     │                      │
│ +reset(seed)→str     │     │ +reset()             │
│ +step(action)→(obs,  │     │ +step()              │
│   done: bool)        │     │ +success()           │
│ +success()→float     │     └──────────────────────┘
└──────────────────────┘
```

---

## Puntos de entrada CLI (comandos disponibles)

### Phase2 (`python -m generate.run_phase2`)
| Subcommand | Descripción | Salida principal |
|------------|-------------|------------------|
| `build-queries` | Genera queries desde fuentes externas | data/queries/queries.parquet |
| `pilot` | Ejecuta ~1000 consultas × configs | data/pilot/responses_*.jsonl |
| `judge` | Etiqueta con juez LLM | pilot/labels_judge.jsonl |
| `audit-export` | Exporta hoja ciega | pilot/audit/audit_sheet.csv |
| `audit-import` | Importa auditoría humana | pilot/labels_human.jsonl |
| `report` | Genera reporte final + puertas | pilot/REPORT.md |

### Phase3 (`python -m generate.run_phase3`)
| Subcommand | Descripción | Salida principal |
|------------|-------------|------------------|
| `init` | Congela subset + split train/test | data/drift/subset/* |
| `new-pool` | Genera pool siguiente con eventos | data/drift/pool_sXX.yaml |
| `snapshot` | Ejecuta inferencias + manifiesto | data/drift/sXX/responses*.jsonl |
| `check` | Verifica comparabilidad | stdout (warnings if any) |
| `report` | Deriva por config + McNemar+Holm | data/drift/DRIFT_REPORT.md |
| `eval` | E4: prequential evaluation | data/drift/e4_results.csv |
| `simulate` | Ensayo sintético completo | stdout (synthetic results) |

### Phase4 (`python -m generate.run_phase4`)
| Subcommand | Descripción | Salida principal |
|------------|-------------|------------------|
| `init` | Tareas + semillas + partición | data/agent/tasks.json |
| `make-prices` | Precios desde configs Fase2/3 | data/agent/prices.json |
| `collect` | Episodios fijos + mixtos | data/agent/episodes.jsonl |
| `stepdata` | Etiquetas contrafactuales | data/agent/stepdata.jsonl |
| `evaluate` | Entrena routers + eval en test | data/agent/eval_episodes.jsonl |
| `analyze` | Reporte H5 + headroom | data/agent/AGENT_REPORT.md |
| `simulate` | Simulación completa mock | data/agent/AGENT_REPORT.md (mock) |

---

*Documentación generada automáticamente el 2026-11-12 desde AST del código fuente.*
