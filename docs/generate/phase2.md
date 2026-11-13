# 📄 phase2.py

**Ruta relativa:** `generate/phase2.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/phase2.py`

## 📖 Descripción del módulo

Fase 2: piloto, análisis de coste real, etiquetado con juez, auditoría humana y puertas de paso.

Medición clave: `latency_s` que guarda VLLMBackend es el tiempo de pared del lote dividido por el
nº de peticiones; por tanto suma(out_tokens)/suma(latency_s) = throughput AGREGADO de decodificación
del servidor (el que necesita pricing.derive_selfhost_prices y estimate_budget).


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from backends import GenRequest`
- `from judge import JUDGE_PROMPT, parse_score, judge_vs_human`
- `from routedata.stats import cohen_kappa`
- `import hashlib`
- `import json`
- `import numpy`
- `import os`
- `import pandas`

## ⚙️ Funciones globales (8)

### 📝 `analyze_pilot(queries, configs, responses, n_full, tp_by_model, gpu_usd_hr, util)`

```python
    Por configuración: acierto, tokens, truncados, cumplimiento de longitud, tok/s agregado y
    proyección de GPU-horas/USD a `n_full` consultas. `tp_by_model`: nº de GPUs por modelo.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `queries` | `Any` | — |
| `configs` | `Any` | — |
| `responses` | `Any` | — |
| `n_full` | `Any` | — |
| `tp_by_model` | `Any` | — |
| `gpu_usd_hr` | `Any` | — |
| `util` | `Any` | — |

**Retorna:** `Any`

### 📝 `gates(report, judge_metrics, big_models, min_compliance, min_pearson, min_kappa, max_unlabeled)`

```python
    Puertas de paso de la Fase 2 (ver PLAN). Devuelve {criterio: (ok, detalle)}.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `report` | `Any` | — |
| `judge_metrics` | `Any` | — |
| `big_models` | `Any` | — |
| `min_compliance` | `Any` | — |
| `min_pearson` | `Any` | — |
| `min_kappa` | `Any` | — |
| `max_unlabeled` | `Any` | — |

**Retorna:** `Any`

### 📝 `apply_judges(queries, responses, judge_backends, batch, seed)`

```python
    Puntúa con un conjunto de jueces las respuestas con label_source=='pending_judge'.
    Devuelve una *capa de etiquetas* (no modifica los shards): columnas query_id, config_id, run_id,
    correct (media de jueces), judge_std, label_source='judge'.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `queries` | `Any` | — |
| `responses` | `Any` | — |
| `judge_backends` | `Any` | — |
| `batch` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `overlay_labels(responses, labels)`

```python
    Aplica una capa de etiquetas (juez o humana) sobre `responses` sin tocar los shards.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `responses` | `Any` | — |
| `labels` | `Any` | — |

**Retorna:** `Any`

### 📝 `export_audit(responses, queries, labels_judge, n, seed, out_dir)`

```python
    Hoja CSV CIEGA (sin modelo/config/etiqueta del juez) + mapa privado item_id -> (query, config).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `responses` | `Any` | — |
| `queries` | `Any` | — |
| `labels_judge` | `Any` | — |
| `n` | `Any` | — |
| `seed` | `Any` | — |
| `out_dir` | `Any` | — |

**Retorna:** `Any`

### 📝 `import_audit(sheet_path, key_path, thr)`

```python
    Lee la hoja anotada: κ entre anotadores, y acuerdo juez–humano (media de humanos).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `sheet_path` | `Any` | — |
| `key_path` | `Any` | — |
| `thr` | `Any` | — |

**Retorna:** `Any`

### 📝 `human_label_layer(merged)`

```python
    Capa de etiquetas humanas (prevalece sobre juez) para los ítems auditados.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `merged` | `Any` | — |

**Retorna:** `Any`

### 📝 `render_report_md(report, gate_res)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `report` | `Any` | — |
| `gate_res` | `Any` | — |

**Retorna:** `str`
