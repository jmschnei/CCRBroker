# 📄 driftroute.py

**Ruta relativa:** `generate/driftroute.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/driftroute.py`

## 📖 Descripción del módulo

DriftRoute: deriva temporal con modelos de pesos abiertos.

Qué *sí* puede derivar con pesos abiertos y pin de revisión (fuentes de deriva):
  (a) nuevas versiones/lanzamientos de modelos (cambio de `revision` o de familia),
  (b) cambios de cuantización, motor de inferencia (versión de vLLM), hardware o semilla,
  (c) cambios de precio (alquiler de GPU, precios de proveedores de los mismos pesos),
  (d) altas y bajas de modelos en el pool,
  (e) deriva de *consultas*: lotes nuevos posteriores al corte (p. ej. problemas de concurso).
Qué NO hay con pesos abiertos: cambios silenciosos de un endpoint propietario. Limitación a
declarar en el artículo (el benchmark mide deriva de pool, no de API cerrada).
Cada snapshot guarda un manifiesto para reproducir y para auditar qué cambió.


## 🔗 Dependencias (imports)

- `from  import configroute, phase2`
- `from  import prompts, efforts`
- `from __future__ import annotations`
- `from dataclasses import dataclass, asdict, field`
- `from routedata.drift import Snap, profile_vector, featurize_tfidf`
- `from routedata.schema import load_table, to_matrices`
- `from routedata.splits import iid_split`
- `from routedata.stats import mcnemar_exact`
- `import copy`
- `import hashlib`
- `import importlib.metadata`
- `import json`
- `import numpy`
- `import os`
- `import pandas`
- `import platform`
- `import re`
- `import subprocess`
- `import time`

## 📦 Clases (1)

### 🧊 `SnapshotManifest`

**Métodos:**

#### 📝 `save(self, d)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `d` | `Any` | — |

---

## ⚙️ Funciones globales (10)

### 📝 `drift_report(resp_a, resp_b, config_col)`

```python
    Compara dos snapshots por config: Δ precisión, Δ tokens de salida y flips (McNemar).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `resp_a` | `Any` | — |
| `resp_b` | `Any` | — |
| `config_col` | `Any` | — |

**Retorna:** `Any`

### 📝 `apply_events(configs, events)`

```python
    Aplica eventos de pool/precio a una tabla `configs` (para simulaciones y replays).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `configs` | `Any` | — |
| `events` | `Any` | — |

**Retorna:** `Any`

### 📝 `capture_environment()`

```python
    Versiones relevantes de la pila de inferencia y GPU (para atribuir deriva a motor/hardware).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|

**Retorna:** `Any`

### 📝 `ids_hash(ids)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `ids` | `Any` | — |

**Retorna:** `Any`

### 📝 `prompt_hash()`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|

**Retorna:** `Any`

### 📝 `comparability_check(manifests)`

```python
    Devuelve avisos si los snapshots no son comparables (consultas, prompts, parámetros o juez
    distintos). Un aviso significa que la 'deriva' medida puede ser un artefacto del protocolo.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `manifests` | `Any` | — |

**Retorna:** `Any`

### 📝 `freeze_subset(queries, n, seed, verifiable_only, exclude_ids, strat)`

```python
    Congela el subconjunto de DriftRoute: estratificado, sin la partición privada, y (opcional)
    solo con tareas verificables para no depender del juez de la Fase 2.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `queries` | `Any` | — |
| `n` | `Any` | — |
| `seed` | `Any` | — |
| `verifiable_only` | `Any` | — |
| `exclude_ids` | `Any` | — |
| `strat` | `Any` | — |

**Retorna:** `str`

### 📝 `split_train_test(queries, test_frac, seed)`

```python
    Partición FIJA por cluster (sin casi-duplicados repartidos). No cambiar entre snapshots.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `queries` | `Any` | — |
| `test_frac` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `apply_pool_events(pool, snapshot_id, add, retire, revision, price_factor, new_batch)`

```python
    Aplica eventos a un dict de pool (para `new-pool`). Devuelve (pool_nuevo, eventos).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `pool` | `Any` | — |
| `snapshot_id` | `Any` | — |
| `add` | `Any` | — |
| `retire` | `Any` | — |
| `revision` | `Any` | — |
| `price_factor` | `Any` | — |
| `new_batch` | `Any` | — |

**Retorna:** `Any`

### 📝 `load_snaps(drift_dir, featurize, dim)`

```python
    Lee snapshots (s00, s01, ...) del disco y construye `routedata.drift.Snap` comparables.
    La featurización se calcula UNA vez sobre la unión de consultas (representación idéntica en todos).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `drift_dir` | `Any` | — |
| `featurize` | `Any` | — |
| `dim` | `Any` | — |

**Retorna:** `str`
