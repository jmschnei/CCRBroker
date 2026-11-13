# 📄 splits.py

**Ruta relativa:** `routedata/splits.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/splits.py`

## 📖 Descripción del módulo

Particiones de evaluación (E1–E8): en distribución, por clusters (anti-fuga),
leave-one-domain-out, leave-one-model-out y temporal.

## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from sklearn.cluster import KMeans`
- `import hashlib`
- `import numpy`
- `import re`

## ⚙️ Funciones globales (8)

### 📝 `normalized_hash(text)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `Any` | — |

**Retorna:** `Any`

### 📝 `dedup_groups(texts)`

```python
    Grupo = hash del texto normalizado (casi-duplicados triviales). Para duplicados
    semánticos usa `cluster_groups` con embeddings.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `texts` | `Any` | — |

**Retorna:** `Any`

### 📝 `cluster_groups(embeddings, k, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `embeddings` | `Any` | — |
| `k` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `iid_split(n, test_frac, seed, groups)`

```python
    70/30 por defecto (como LLMRouterBench). Con `groups`, ningún grupo se reparte entre
    train y test (split por cluster).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `n` | `Any` | — |
| `test_frac` | `Any` | — |
| `seed` | `Any` | — |
| `groups` | `Any` | — |

**Retorna:** `Any`

### 📝 `leave_group_out(groups)`

```python
    Itera (nombre, idx_train, idx_test) dejando fuera un dominio/dataset cada vez.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `groups` | `Any` | — |

**Retorna:** `Any`

### 📝 `leave_model_out(n_models, n_holdout, n_splits, seed)`

```python
    Subconjuntos de modelos 'no vistos' para medir generalización a modelos nuevos.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `n_models` | `Any` | — |
| `n_holdout` | `Any` | — |
| `n_splits` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `temporal_split(timestamps, cutoff)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `timestamps` | `Any` | — |
| `cutoff` | `Any` | — |

**Retorna:** `Any`

### 📝 `prequential(snapshot_ids)`

```python
    Para snapshots ordenados s0<s1<...: entrena con ≤si, evalúa en s(i+1).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `snapshot_ids` | `Any` | — |

**Retorna:** `Any`
