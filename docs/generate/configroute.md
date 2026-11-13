# 📄 configroute.py

**Ruta relativa:** `generate/configroute.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/configroute.py`

## 📖 Descripción del módulo

ConfigRoute: respuestas por configuración (modelo × esfuerzo × presupuesto de longitud).

Flujo (modelo-a-modelo para caber en GPU; reanudable):
  1) sample_queries()  → muestra estratificada por dominio/dificultad
  2) build_config_grid() → tabla `configs`
  3) run()  → para cada modelo: carga backend, recorre sus configs, genera, etiqueta, guarda shards
Etiquetado: verificadores programáticos; las tareas abiertas quedan con correct=NaN y
label_source='pending_judge' hasta pasar por judge.py.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from backends import GenRequest`
- `from efforts import EFFORTS, run_with_effort, EffortSpec`
- `from prompts import build_messages`
- `from routedata.schema import save_table, load_table, validate`
- `from verifiers import verify`
- `import itertools`
- `import json`
- `import numpy`
- `import os`
- `import pandas`
- `import time`

## ⚙️ Funciones globales (5)

### 📝 `sample_queries(sources, quotas, strat, seed)`

```python
    sources: {nombre: DataFrame de queries}; quotas: {nombre: n}. Muestreo estratificado
    proporcional por `strat` (compatible con pandas>=3: sin groupby.apply).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `sources` | `Any` | — |
| `quotas` | `Any` | — |
| `strat` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `str`

### 📝 `build_config_grid(models, budgets, efforts, hardware, price_date)`

```python
    models: {model: {'revision':..., 'price_in':..., 'price_out':..., 'thinking': bool, 'price_source':...}}.
    Solo los modelos con `thinking=True` reciben los niveles think_*; el resto solo 'nothink'.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `models` | `Any` | — |
| `budgets` | `Any` | — |
| `efforts` | `Any` | — |
| `hardware` | `Any` | — |
| `price_date` | `Any` | — |

**Retorna:** `bool`

### 📝 `_done_keys(out_dir)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `out_dir` | `Any` | — |

**Retorna:** `Any`

### 📝 `run(queries, configs, backend_factory, out_dir, run_id, snapshot_id, batch, seed, log)`

```python
    backend_factory(model, revision) -> Backend. Reanuda saltando (query, config, run) ya hechos.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `queries` | `Any` | — |
| `configs` | `Any` | — |
| `backend_factory` | `Any` | — |
| `out_dir` | `Any` | — |
| `run_id` | `Any` | — |
| `snapshot_id` | `Any` | — |
| `batch` | `Any` | — |
| `seed` | `Any` | — |
| `log` | `Any` | — |

**Retorna:** `Any`

### 📝 `load_responses(out_dir)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `out_dir` | `Any` | — |

**Retorna:** `Any`
