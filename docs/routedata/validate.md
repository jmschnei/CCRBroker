# 📄 validate.py

**Ruta relativa:** `routedata/validate.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/validate.py`

## 📖 Descripción del módulo


Validación de datasets de LLM routing.

Verifica la integridad y estructura de los datos descargados en data/raw/:
  - RouterBench: data/raw/routerbench/*.pkl
  - SPROUT:      data/raw/sprout/*.parquet
  - LLMRouterBench: data/raw/llmrouterbench/extracted/**/*.json
  - R2-Bench:    data/raw/r2bench/data/*/*/*.csv

Uso:
    from routedata.validate import validate_all
    reports = validate_all()
    for name, report in reports.items():
        print(f"{name}: {'OK' if report['valid'] else 'FAIL'}")


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from collections import defaultdict`
- `from dataclasses import dataclass, field`
- `from pathlib import Path`
- `from typing import Any, Dict, List, Optional`
- `import argparse`
- `import json`
- `import logging`
- `import pandas`
- `import pickle`

## 📦 Clases (1)

### 🧊 `ValidationReport`

Resultado de la validación de un dataset.

**Métodos:**

#### 📝 `to_dict(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

---

## ⚙️ Funciones globales (8)

### 📝 `validate_routerbench(data_root)`

```python
    Valida los archivos .pkl de RouterBench.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `data_root` | `Any` | — |

**Retorna:** `Any`

### 📝 `validate_sprout(data_root)`

```python
    Valida los archivos .parquet de SPROUT.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `data_root` | `Any` | — |

**Retorna:** `Any`

### 📝 `validate_llmrouterbench(data_root)`

```python
    Valida los archivos .json de LLMRouterBench.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `data_root` | `Any` | — |

**Retorna:** `Any`

### 📝 `validate_r2bench(data_root)`

```python
    Valida los archivos .csv de R2-Bench.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `data_root` | `Any` | — |

**Retorna:** `Any`

### 📝 `validate_all(data_root)`

```python
    Validates all datasets and returns a dict of reports.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `data_root` | `Any` | — |

**Retorna:** `Any`

### 📝 `print_summary(reports)`

```python
    Prints a human-readable summary of validation results.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `reports` | `Any` | — |

**Retorna:** `Any`

### 📝 `validate_unified(data_root)`

```python
    Validate processed unified Parquet tables against their manifest and each other.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `data_root` | `Any` | — |

**Retorna:** `Any`

### 📝 `main(argv)`

```python
    CLI entry point.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `argv` | `Any` | — |

**Retorna:** `Any`
