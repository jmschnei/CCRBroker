# 📄 judge.py

**Ruta relativa:** `generate/judge.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/judge.py`

## 📖 Descripción del módulo

Juez LLM (conjunto) para tareas abiertas + muestreo para anotación humana.

Con solo pesos abiertos: usa ≥3 jueces de familias distintas y, si es posible, que NINGUNO
sea también un modelo del pool evaluado (sesgo de autopreferencia). Valida SIEMPRE contra
una muestra humana (objetivo: Pearson ≥ 0,8 y κ ≥ 0,7 tras binarizar), como hace R2-Bench
(500 respuestas, 30 anotadores; Pearson 0,82 con el juez elegido).

## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from backends import GenRequest`
- `from routedata.stats import cohen_kappa`
- `import json`
- `import numpy`
- `import re`

## ⚙️ Funciones globales (4)

### 📝 `parse_score(text)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `Any` | — |

**Retorna:** `Any`

### 📝 `judge_ensemble(judge_backends, items, seed)`

```python
    items: lista de dicts con question/reference/response. Devuelve (media, desviación) por ítem.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `judge_backends` | `Any` | — |
| `items` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `sample_for_human(df, n, strata, seed)`

```python
    Muestra estratificada para anotación humana (≥2 anotadores por ítem recomendados).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `Any` | — |
| `n` | `Any` | — |
| `strata` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `str`

### 📝 `judge_vs_human(judge, human, thr)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `judge` | `Any` | — |
| `human` | `Any` | — |
| `thr` | `Any` | — |

**Retorna:** `Any`
