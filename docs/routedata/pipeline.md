# 📄 pipeline.py

**Ruta relativa:** `routedata/pipeline.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/pipeline.py`

## 📖 Descripción del módulo

Pipeline: carga todos los datasets y genera el esquema unificado.

Uso:
    python -m routedata.pipeline load
    python -m routedata.pipeline load --output data/processed/unified/


## 🔗 Dependencias (imports)

- `from adapters.llmrouterbench import LLMRouterBenchAdapter`
- `from adapters.r2bench import R2BenchAdapter`
- `from adapters.routerbench import load_both`
- `from adapters.sprout import SproutAdapter, create_sprout_adapter`
- `from pathlib import Path`
- `import argparse`
- `import json`
- `import pandas`
- `import sys`
- `import time`

## 📦 Clases (1)

### 🧊 `Pipeline`

Carga todos los datasets y genera el esquema unificado.

**Métodos:**

#### 📝 `__init__(self, data_root)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `data_root` | `Any` | — |

#### 📝 `load_all(self)`

```python
    Carga todos los datasets y devuelve (queries, configs, responses) por dataset.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

#### 📝 `save(self, results, output_dir)`

```python
    Guarda los resultados en formato Parquet.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `results` | `Any` | — |
| `output_dir` | `Any` | — |

#### 📝 `run(self, output_dir)`

```python
    Ejecuta el pipeline completo.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `output_dir` | `Any` | — |

---

## ⚙️ Funciones globales (1)

### 📝 `main()`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|

**Retorna:** `Any`
