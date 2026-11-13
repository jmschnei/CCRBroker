# 📄 run_phase3.py

**Ruta relativa:** `generate/run_phase3.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/run_phase3.py`

## 📖 Descripción del módulo

CLI de la Fase 3 (DriftRoute).  Estructura de datos:

  data/drift/subset/{queries.*, split.json, subset_manifest.json}   <- congelado en `init`
  data/drift/s00/, s01/, ...  (configs, responses_*.jsonl, manifest_sXX.json, labels_*.jsonl opcionales)

Subcomandos:
  init      congela subconjunto + partición train/test (cluster)
  new-pool  genera el pool del siguiente snapshot aplicando eventos (y los registra)
  snapshot  ejecuta un snapshot (reanudable) + manifiesto + reprecio por throughput medido
  check     verifica comparabilidad entre snapshots
  report    deriva por config entre snapshots consecutivos (McNemar + Holm)
  eval      E4: evaluación prequential de routers (estático / reentrenado / perfil+online)
  simulate  ensayo en seco con datos sintéticos (sin GPU)


## 🔗 Dependencias (imports)

- `from  import configroute, driftroute`
- `from __future__ import annotations`
- `from backends import MockBackend`
- `from backends import VLLMBackend`
- `from routedata import pricing, drift`
- `from routedata.schema import save_table, load_table`
- `from routedata.stats import holm`
- `import argparse`
- `import hashlib`
- `import json`
- `import numpy`
- `import os`
- `import pandas`
- `import re`
- `import yaml`

## ⚙️ Funciones globales (12)

### 📝 `_mock_factory(pool)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `pool` | `Any` | — |

**Retorna:** `Any`

### 📝 `_factory(kind, pool)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `kind` | `Any` | — |
| `pool` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_init(a, cfg)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `cfg` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_new_pool(a, cfg)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `cfg` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_snapshot(a, cfg)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `cfg` | `Any` | — |

**Retorna:** `Any`

### 📝 `_manifests(drift_dir)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `drift_dir` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_check(a, cfg)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `cfg` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_report(a, cfg)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `cfg` | `Any` | — |

**Retorna:** `Any`

### 📝 `_run_eval(snaps, train_ids, test_ids, cfg, drift_dir)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `snaps` | `Any` | — |
| `train_ids` | `Any` | — |
| `test_ids` | `Any` | — |
| `cfg` | `Any` | — |
| `drift_dir` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_eval(a, cfg)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `cfg` | `Any` | — |

**Retorna:** `Any`

### 📝 `cmd_simulate(a, cfg)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `cfg` | `Any` | — |

**Retorna:** `Any`

### 📝 `main(argv)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `argv` | `Any` | — |

**Retorna:** `Any`
