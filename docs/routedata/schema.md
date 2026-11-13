# 📄 schema.py

**Ruta relativa:** `routedata/schema.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/schema.py`

## 📖 Descripción del módulo

Esquema unificado de datos para routing.

Tres tablas (formato largo, una fila por entidad):
  queries   : una fila por consulta
  configs   : una fila por configuración (modelo, esfuerzo, presupuesto)
  responses : una fila por (query_id, config_id, run_id)

Se guardan en parquet si pyarrow está disponible; si no, en JSONL.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from typing import Iterable`
- `import json`
- `import numpy`
- `import os`
- `import pandas`
- `import pyarrow`

## ⚙️ Funciones globales (5)

### 📝 `_has_parquet()`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|

**Retorna:** `Any`

### 📝 `save_table(df, path_noext)`

```python
    Guarda en parquet (si hay pyarrow) o JSONL. Devuelve la ruta escrita.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `Any` | — |
| `path_noext` | `Any` | — |

**Retorna:** `Any`

### 📝 `load_table(path_noext)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `path_noext` | `Any` | — |

**Retorna:** `Any`

### 📝 `validate(df, kind)`

```python
    Devuelve lista de problemas (vacía si todo OK).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `Any` | — |
| `kind` | `Any` | — |

**Retorna:** `Any`

### 📝 `to_matrices(queries, configs, responses, run_id)`

```python
    Convierte tablas largas en matrices Q[n,k] (calidad) y C[n,k] (coste USD).

    Las celdas sin respuesta quedan como NaN; usa `mask` para filtrar.
    Coste = (in_tokens*price_in + (out_tokens)*price_out)/1e6, donde out_tokens
    ya incluye los tokens de razonamiento (se facturan como salida).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `queries` | `Any` | — |
| `configs` | `Any` | — |
| `responses` | `Any` | — |
| `run_id` | `Any` | — |

**Retorna:** `Any`
