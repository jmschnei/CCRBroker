# 📄 agent.py

**Ruta relativa:** `routedata/agent.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/agent.py`

## 📖 Descripción del módulo

Routers y evaluación para agentes (Fase 4 · RQ5 · H5).

TaskRouter  : elige UN modelo por tarea a partir de la descripción inicial (baseline 'por tarea').
StepRouter  : elige el modelo en CADA paso con utilidades contrafactuales aprendidas del estado actual
              (observación + paso) y un coste de paso EXACTO calculado con el modelo de caché; el coste
              de cambio de modelo aparece como prefill extra no cacheado (penalty=True) o se ignora (False).
Evaluación  : éxito/coste/latencia por episodio recalculables bajo distintos modos de caché y precios;
              comparaciones pareadas por tarea (bootstrap, McNemar, Holm).


## 🔗 Dependencias (imports)

- `from  import metrics`
- `from __future__ import annotations`
- `from generate.agentroute import candidate_prefill, approx_context, steps_cost, Episode`
- `from sklearn.feature_extraction.text import HashingVectorizer`
- `from sklearn.linear_model import Ridge`
- `from sklearn.neighbors import NearestNeighbors`
- `from stats import mcnemar_exact, holm, bootstrap_ci`
- `import numpy`
- `import pandas`

## 📦 Clases (2)

### 🧊 `TaskRouter`

**Métodos:**

#### 📝 `__init__(self, k)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `k` | `Any` | — |

#### 📝 `fit(self, X, S, C)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `X` | `Any` | — |
| `S` | `Any` | — |
| `C` | `Any` | — |

#### 📝 `predict(self, x)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `x` | `Any` | — |

---

### 🧊 `StepRouter`

Ridge por modelo sobre rasgos del estado: utilidad esperada y tokens de salida esperados.

**Métodos:**

#### 📝 `__init__(self, models, alpha)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `models` | `Any` | — |
| `alpha` | `Any` | — |

#### 📝 `fit(self, rows, max_steps)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `rows` | `Any` | — |
| `max_steps` | `Any` | — |

#### 📝 `predict(self, obs, t)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `obs` | `Any` | — |
| `t` | `Any` | — |

---

## ⚙️ Funciones globales (8)

### 📝 `featurize_text(texts, n_features)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `texts` | `Any` | — |
| `n_features` | `Any` | — |

**Retorna:** `Any`

### 📝 `featurize_state(obs, t, max_steps)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `obs` | `Any` | — |
| `t` | `Any` | — |
| `max_steps` | `Any` | — |

**Retorna:** `Any`

### 📝 `task_router_policy(router, models, lam, featurize)`

```python
    Elige un modelo al inicio: argmax Ŝ − λ·Ĉ_USD y lo mantiene.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `router` | `Any` | — |
| `models` | `Any` | — |
| `lam` | `Any` | — |
| `featurize` | `Any` | — |

**Retorna:** `Any`

### 📝 `step_router_policy(router, price, lam, cache_mode, retention, penalty, stick_bonus_usd)`

```python
    Score(m) = Û(m) − λ·[coste_paso_exacto(m) + Ĉ_posterior(m)] (+ bonus de permanencia). Con penalty=True, coste_paso usa el prefill REAL no
    cacheado de m (cambiar de modelo lo encarece); con penalty=False se ignora el estado de caché (ingenuo).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `router` | `Any` | — |
| `price` | `Any` | — |
| `lam` | `Any` | — |
| `cache_mode` | `Any` | — |
| `retention` | `Any` | — |
| `penalty` | `Any` | — |
| `stick_bonus_usd` | `Any` | — |

**Retorna:** `Any`

### 📝 `episodes_to_frame(episodes, price, cache_modes, retention)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `episodes` | `Any` | — |
| `price` | `Any` | — |
| `cache_modes` | `Any` | — |
| `retention` | `Any` | — |

**Retorna:** `Any`

### 📝 `summarize(df, cache_modes, n_boot, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `Any` | — |
| `cache_modes` | `Any` | — |
| `n_boot` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `compare_policies(df, base, others, cache_mode, n_boot, seed)`

```python
    Diferencias pareadas por tarea frente a `base`: Δéxito y Δcoste (IC bootstrap), McNemar sobre éxito y Holm.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `Any` | — |
| `base` | `Any` | — |
| `others` | `Any` | — |
| `cache_mode` | `Any` | — |
| `n_boot` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `str`

### 📝 `headroom(df_fixed, cache_mode)`

```python
    ¿Hay margen para enrutar? Éxito del mejor modelo fijo vs oráculo POR TAREA (el más barato que triunfa).
    Si el oráculo por tarea no supera al mejor fijo, el routing por tarea no puede ayudar (el por paso puede).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df_fixed` | `Any` | — |
| `cache_mode` | `Any` | — |

**Retorna:** `Any`
