# 📄 run_phase2.py

**Ruta relativa:** `generate/run_phase2.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/run_phase2.py`

## 📖 Descripción del módulo

CLI de la Fase 2.  Uso (desde la raíz del kit):

  python -m generate.run_phase2 build-queries --cfg configs/configroute_pilot.yaml --out data/queries [--open-ended abiertas.jsonl]
  python -m generate.run_phase2 pilot   --pool configs/pool_openweights.yaml --cfg configs/configroute_pilot.yaml --queries data/queries --out data/pilot [--backend vllm|mock] [--models m1,m2]
  python -m generate.run_phase2 judge   --run-dir data/pilot --pool configs/pool_openweights.yaml [--backend vllm|mock]
  python -m generate.run_phase2 audit-export --run-dir data/pilot --n 150
  python -m generate.run_phase2 audit-import --run-dir data/pilot --sheet data/pilot/audit/audit_sheet_filled.csv
  python -m generate.run_phase2 report  --run-dir data/pilot --pool configs/pool_openweights.yaml --cfg configs/configroute_pilot.yaml


## 🔗 Dependencias (imports)

- `from  import configroute, phase2, build_queries`
- `from __future__ import annotations`
- `from backends import MockBackend, GenResult`
- `from backends import VLLMBackend`
- `from routedata.schema import save_table, load_table`
- `import argparse`
- `import json`
- `import numpy`
- `import os`
- `import pandas`
- `import sys`
- `import yaml`

## 📦 Clases (1)

### 🧊 `_JudgeMock`

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `name` | — |

**Métodos:**

#### 📝 `generate(self, reqs)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `reqs` | `Any` | — |

---

## ⚙️ Funciones globales (3)

### 📝 `_pool(path)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `path` | `Any` | — |

**Retorna:** `Any`

### 📝 `_factory(kind, pool)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `kind` | `Any` | — |
| `pool` | `Any` | — |

**Retorna:** `Any`

### 📝 `main(argv)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `argv` | `Any` | — |

**Retorna:** `Any`
