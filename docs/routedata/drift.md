# 📄 drift.py

**Ruta relativa:** `routedata/drift.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/drift.py`

## 📖 Descripción del módulo

E4 (RQ4 · H4): routers frente a deriva del pool y evaluación *prequential* por snapshots.

Protocolo (evita fugas):
  - Las consultas del subconjunto DriftRoute se dividen UNA vez por cluster en `train_ids` y `test_ids`
    y la división no cambia entre snapshots. Los lotes nuevos (deriva de consultas) van solo a test.
  - En cada snapshot s, el router puede adaptarse con una fracción f de las consultas de train_ids
    *con feedback real en s* ('tráfico de calibración', evaluado en sombra sobre todas las configs).
  - Se evalúa siempre sobre test_ids en s.
Routers: StaticRouter (entrenado en s0, nunca actualizado), RefitRouter(f) (kNN reentrenado en s con la
fracción f; f=1 es la cota superior de 'reentrenar'), ProfileOnlineRouter(f) (priores por perfil de
configuración para configs nuevas + corrección online de sesgo de calidad y escala de coste).
Métricas: AIQ relativa al techo (oráculo + configs individuales), pérdida relativa del estático frente
al reentrenado, y fracción de tráfico necesaria para recuperar el 80 % de la pérdida.


## 🔗 Dependencias (imports)

- `from  import metrics`
- `from __future__ import annotations`
- `from dataclasses import dataclass, field`
- `from sklearn.decomposition import TruncatedSVD`
- `from sklearn.feature_extraction.text import TfidfVectorizer`
- `from sklearn.neighbors import NearestNeighbors`
- `import numpy`
- `import pandas`
- `import warnings`

## 📦 Clases (4)

### 🧊 `Snap`

---

### 🧊 `StaticRouter`

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `name` | — |

**Métodos:**

#### 📝 `__init__(self, k)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `k` | `Any` | — |

#### 📝 `fit(self, s0, train_ids)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `s0` | `Any` | — |
| `train_ids` | `Any` | — |

#### 📝 `predict(self, snap, train_ids, f, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `snap` | `Any` | — |
| `train_ids` | `Any` | — |
| `f` | `Any` | — |
| `seed` | `Any` | — |

---

### 🧊 `RefitRouter`

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `name` | — |

**Métodos:**

#### 📝 `__init__(self, k)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `k` | `Any` | — |

#### 📝 `fit(self, s0, train_ids)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `s0` | `Any` | — |
| `train_ids` | `Any` | — |

#### 📝 `predict(self, snap, train_ids, f, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `snap` | `Any` | — |
| `train_ids` | `Any` | — |
| `f` | `Any` | — |
| `seed` | `Any` | — |

---

### 🧊 `ProfileOnlineRouter`

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `name` | — |

**Métodos:**

#### 📝 `__init__(self, k, n_prior, tau, temp)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `k` | `Any` | — |
| `n_prior` | `Any` | — |
| `tau` | `Any` | — |
| `temp` | `Any` | — |

#### 📝 `fit(self, s0, train_ids)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `s0` | `Any` | — |
| `train_ids` | `Any` | — |

#### 📝 `predict(self, snap, train_ids, f, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `snap` | `Any` | — |
| `train_ids` | `Any` | — |
| `f` | `Any` | — |
| `seed` | `Any` | — |

---

## ⚙️ Funciones globales (9)

### 📝 `_knn(Xtr, Ytr, Xte, k)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Xtr` | `Any` | — |
| `Ytr` | `Any` | — |
| `Xte` | `Any` | — |
| `k` | `Any` | — |

**Retorna:** `Any`

### 📝 `_rows(snap, ids)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `snap` | `Any` | — |
| `ids` | `Any` | — |

**Retorna:** `Any`

### 📝 `_curve(Qh, Ch, Q, C, rows, lams)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Qh` | `Any` | — |
| `Ch` | `Any` | — |
| `Q` | `Any` | — |
| `C` | `Any` | — |
| `rows` | `Any` | — |
| `lams` | `Any` | — |

**Retorna:** `Any`

### 📝 `evaluate_snapshot(router, snap, train_ids, test_ids, f, seed, lams, subset)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `router` | `Any` | — |
| `snap` | `Any` | — |
| `train_ids` | `Any` | — |
| `test_ids` | `Any` | — |
| `f` | `Any` | — |
| `seed` | `Any` | — |
| `lams` | `Any` | — |
| `subset` | `Any` | — |

**Retorna:** `Any`

### 📝 `prequential(snaps, train_ids, test_ids, fractions, seed, subset)`

```python
    Devuelve DataFrame: snapshot, evento(s), router, f, aiq, norm_gain. Entrena en snaps[0].
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `snaps` | `Any` | — |
| `train_ids` | `Any` | — |
| `test_ids` | `Any` | — |
| `fractions` | `Any` | — |
| `seed` | `Any` | — |
| `subset` | `Any` | — |

**Retorna:** `Any`

### 📝 `h4_summary(df, target)`

```python
    H4: pérdida relativa de AIQ del estático frente al reentrenado completo, y fracción mínima de
    tráfico con la que cada router recupera `target` de esa pérdida (en norm_gain).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `Any` | — |
| `target` | `Any` | — |

**Retorna:** `Any`

### 📝 `featurize_tfidf(texts, dim, seed)`

```python
    Rasgos baratos (TF-IDF + SVD) para pruebas. En producción usa un modelo de embeddings fijo y
    NO lo cambies entre snapshots (cambiaría la representación y confundiría la deriva).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `texts` | `Any` | — |
| `dim` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `profile_vector(model_size_b, thinking, effort, budget, price_out)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `model_size_b` | `Any` | — |
| `thinking` | `Any` | — |
| `effort` | `Any` | — |
| `budget` | `Any` | — |
| `price_out` | `Any` | — |

**Retorna:** `Any`

### 📝 `simulate_drift(n, seed, n_new)`

```python
    Secuencia de snapshots sintética con eventos: s1 regresión de revisión, s2 modelo nuevo,
    s3 bajada de precio, s4 baja de un modelo, s5 lote de consultas nuevas (desplazadas).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `n` | `Any` | — |
| `seed` | `Any` | — |
| `n_new` | `Any` | — |

**Retorna:** `Any`
