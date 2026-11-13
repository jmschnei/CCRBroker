# 📄 baselines.py

**Ruta relativa:** `routedata/baselines.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/baselines.py`

## 📖 Descripción del módulo

Baselines obligatorios (triviales) y un router kNN de referencia para validar el pipeline.

## 🔗 Dependencias (imports)

- `from  import metrics`
- `from __future__ import annotations`
- `from sklearn.neighbors import NearestNeighbors`
- `import numpy`

## 📦 Clases (1)

### 🧊 `KNNRouter`

Router de referencia: promedia Q y C de los k vecinos (coseno) en el espacio de
embeddings. Sirve de baseline fuerte-y-simple y para probar el pipeline.

**Métodos:**

#### 📝 `__init__(self, k)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `k` | `Any` | — |

#### 📝 `fit(self, X, Q, C)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `X` | `Any` | — |
| `Q` | `Any` | — |
| `C` | `Any` | — |

#### 📝 `predict(self, X)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `X` | `Any` | — |

---

## ⚙️ Funciones globales (1)

### 📝 `trivial_baselines(Q, C, groups, seed)`

```python
    Devuelve {nombre: vector de elecciones}. 'best_single' y 'oracle' son a posteriori
    (cotas de referencia, no routers desplegables).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q` | `Any` | — |
| `C` | `Any` | — |
| `groups` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`
