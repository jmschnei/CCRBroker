# 📄 run_phase4.py

**Ruta relativa:** `generate/run_phase4.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/run_phase4.py`

## 📖 Descripción del módulo

CLI de la Fase 4 (AgentRoute).  Datos en data/agent/:
  tasks.json         entorno, semillas de tareas y partición train/test (por tarea)
  prices.json        {modelo: {"in","out" USD/Mtok, "prefill_tps","decode_tps"}}   (de `make-prices`)
  episodes.jsonl     episodios con políticas fijas (todas las tareas) y mixtas (train)
  stepdata.jsonl     etiquetas contrafactuales por (estado, modelo) en tareas de train
  eval_episodes.jsonl, eval_meta.json, AGENT_REPORT.md

Subcomandos: init · make-prices · collect · stepdata · evaluate · analyze · simulate
Modelos: pásalos de más barato a más caro (--models small,mid,big).


## 🔗 Dependencias (imports)

- `from  import agentroute`
- `from __future__ import annotations`
- `from backends import MockBackend, OpenAICompatBackend`
- `from concurrent.futures import ThreadPoolExecutor`
- `from envs import get_env_factory`
- `from routedata import agent_routing`
- `from routedata.schema import load_table`
- `import argparse`
- `import json`
- `import numpy`
- `import os`
- `import pandas`
- `import tempfile`

## ⚙️ Funciones globales (17)

### 📝 `_read(p)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `p` | `Any` | — |

**Retorna:** `Any`

### 📝 `_append(p, rows)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `p` | `Any` | — |
| `rows` | `Any` | — |

**Retorna:** `Any`

### 📝 `_backends(a, models)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `models` | `Any` | — |

**Retorna:** `Any`

### 📝 `_ctx(a)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |

**Retorna:** `Any`

### 📝 `_episode(fac, be, policy, name, seed, max_steps, env_name)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `fac` | `Any` | — |
| `be` | `Any` | — |
| `policy` | `Any` | — |
| `name` | `Any` | — |
| `seed` | `Any` | — |
| `max_steps` | `Any` | — |
| `env_name` | `Any` | — |

**Retorna:** `Any`

### 📝 `_run_many(jobs, workers)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `jobs` | `Any` | — |
| `workers` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_init(a)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_make_prices(a)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_collect(a)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_stepdata(a)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |

**Retorna:** `Any`

### 📝 `_fit(a, d, T, fac, pr, models, fit_tasks)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `d` | `Any` | — |
| `T` | `Any` | — |
| `fac` | `Any` | — |
| `pr` | `Any` | — |
| `models` | `Any` | — |
| `fit_tasks` | `Any` | — |

**Retorna:** `Any`

### 📝 `_policies(tr, sr, models, pr, lam, cache_mode, retention)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `tr` | `Any` | — |
| `sr` | `Any` | — |
| `models` | `Any` | — |
| `pr` | `Any` | — |
| `lam` | `Any` | — |
| `cache_mode` | `Any` | — |
| `retention` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_evaluate(a)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |

**Retorna:** `Any`

### 📝 `cache_model_check(eps, retention)`

```python
    Compara los tokens cacheados que PREDICE el modelo de caché con los que reportó el servidor (si los expone).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `eps` | `Any` | — |
| `retention` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_analyze(a)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_simulate(a)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |

**Retorna:** `Any`

### 📝 `main(argv)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `argv` | `Any` | — |

**Retorna:** `Any`
