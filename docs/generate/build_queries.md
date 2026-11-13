# 📄 build_queries.py

**Ruta relativa:** `generate/build_queries.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/build_queries.py`

## 📖 Descripción del módulo

Construcción de las consultas de ConfigRoute a partir de datasets públicos.

AVISO: los esquemas de columnas de abajo proceden de mi memoria de las fichas de Hugging Face
y NO se han verificado desde este entorno (sin red). Ejecuta `python -m generate.build_queries
inspect <nombre>` en tu máquina y ajusta `map` si difiere. Comprueba también licencia y
condiciones de uso (GPQA es de acceso restringido; no lo redistribuyas).


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from datasets import load_dataset`
- `from routedata.splits import dedup_groups`
- `import json`
- `import numpy`
- `import pandas`
- `import sys`

## ⚙️ Funciones globales (5)

### 📝 `_level(v)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `v` | `Any` | — |

**Retorna:** `Any`

### 📝 `normalize(df, name, spec, seed)`

```python
    DataFrame crudo -> tabla `queries` del esquema unificado.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `Any` | — |
| `name` | `Any` | — |
| `spec` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `load_source(name, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `name` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `drop_duplicates_across(queries)`

```python
    Elimina casi-duplicados exactos entre fuentes (mismo texto normalizado).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `queries` | `Any` | — |

**Retorna:** `Any`

### 📝 `assign_empirical_difficulty(queries, Q, easy, hard)`

```python
    Dificultad empírica = tasa de acierto sobre todas las configs del pool -> easy/medium/hard.
    Úsala tras el piloto para estratificar la muestra completa de forma homogénea entre fuentes.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `queries` | `Any` | — |
| `Q` | `Any` | — |
| `easy` | `Any` | — |
| `hard` | `Any` | — |

**Retorna:** `str`
