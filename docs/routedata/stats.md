# 📄 stats.py

**Ruta relativa:** `routedata/stats.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/stats.py`

## 📖 Descripción del módulo

Estadística para comparar routers: bootstrap por consulta, pruebas pareadas y
corrección por comparaciones múltiples.

## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from math import comb`
- `import numpy`

## ⚙️ Funciones globales (6)

### 📝 `bootstrap_ci(values, stat, n_boot, alpha, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `values` | `Any` | — |
| `stat` | `Any` | — |
| `n_boot` | `Any` | — |
| `alpha` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `paired_bootstrap_diff(a, b, n_boot, alpha, seed)`

```python
    IC de mean(a-b) remuestreando consultas (a y b alineados).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `b` | `Any` | — |
| `n_boot` | `Any` | — |
| `alpha` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `str`

### 📝 `mcnemar_exact(correct_a, correct_b)`

```python
    Prueba exacta de McNemar (binomial bilateral) sobre aciertos binarios pareados.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `correct_a` | `Any` | — |
| `correct_b` | `Any` | — |

**Retorna:** `Any`

### 📝 `holm(pvals, alpha)`

```python
    Corrección de Holm. Devuelve (p_ajustados, rechazos).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `pvals` | `Any` | — |
| `alpha` | `Any` | — |

**Retorna:** `Any`

### 📝 `benjamini_hochberg(pvals, alpha)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `pvals` | `Any` | — |
| `alpha` | `Any` | — |

**Retorna:** `Any`

### 📝 `cohen_kappa(a, b)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `b` | `Any` | — |

**Retorna:** `Any`
