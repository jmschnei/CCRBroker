# 📄 conformal.py

**Ruta relativa:** `routedata/conformal.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/conformal.py`

## 📖 Descripción del módulo

Control de riesgo conformal para CCR (router de configuraciones con garantías).

Política con umbral t: ordena las configs por coste medio (barata→cara); elige la primera
cuya probabilidad predicha de éxito P̂ ≥ t; si ninguna, escala a la más fiable (argmax P̂).
Pérdida por consulta L_i(t) = 1 − Q[i, elegida(t)]. Si L es (aprox.) no creciente en t,
conformal risk control (Angelopoulos et al., 2022) da E[L(t̂)] ≤ α bajo intercambiabilidad:

    t̂ = min{ t : n/(n+1)·R̂_n(t) + B/(n+1) ≤ α },  B = 1.

Nota: la monotonía de L en t NO está garantizada para cualquier predictor; se comprueba
empíricamente (`monotonicity_violation`) y debe reportarse (ver E7).


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `import numpy`

## ⚙️ Funciones globales (6)

### 📝 `choose(Phat, cost_order, t)`

```python
    Elecciones para umbral t. Phat[n,k]; cost_order: índices de configs de barata a cara.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Phat` | `Any` | — |
| `cost_order` | `Any` | — |
| `t` | `Any` | — |

**Retorna:** `Any`

### 📝 `risk_at(Q, Phat, cost_order, t)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q` | `Any` | — |
| `Phat` | `Any` | — |
| `cost_order` | `Any` | — |
| `t` | `Any` | — |

**Retorna:** `Any`

### 📝 `crc_threshold(Q_cal, Phat_cal, cost_order, alpha, grid)`

```python
    Umbral CRC más bajo (= más barato) que cumple el límite de riesgo. None si no existe.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q_cal` | `Any` | — |
| `Phat_cal` | `Any` | — |
| `cost_order` | `Any` | — |
| `alpha` | `Any` | — |
| `grid` | `Any` | — |

**Retorna:** `Any`

### 📝 `monotonicity_violation(Q, Phat, cost_order, grid)`

```python
    Mayor subida de riesgo al aumentar t (0 = perfectamente monótona).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q` | `Any` | — |
| `Phat` | `Any` | — |
| `cost_order` | `Any` | — |
| `grid` | `Any` | — |

**Retorna:** `Any`

### 📝 `stratified_crc(Q_cal, Phat_cal, strata_cal, cost_order, alpha, grid)`

```python
    Un umbral por estrato (dominio/dificultad). Estratos con n pequeño caen al global.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q_cal` | `Any` | — |
| `Phat_cal` | `Any` | — |
| `strata_cal` | `Any` | — |
| `cost_order` | `Any` | — |
| `alpha` | `Any` | — |
| `grid` | `Any` | — |

**Retorna:** `str`

### 📝 `apply_stratified(Phat, strata, thresholds, cost_order, fallback)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Phat` | `Any` | — |
| `strata` | `Any` | — |
| `thresholds` | `Any` | — |
| `cost_order` | `Any` | — |
| `fallback` | `Any` | — |

**Retorna:** `Any`
