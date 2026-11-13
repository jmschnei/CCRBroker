# 📄 multilingual.py

**Ruta relativa:** `generate/multilingual.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/multilingual.py`

## 📖 Descripción del módulo

Subconjuntos ES/DE/FR de ConfigRoute (RQ8): traducción con LLM local + controles + revisión humana.

Reglas: (1) no se traducen consultas de código; (2) las respuestas doradas de mc/math no cambian
(letra/número); (3) cada traducción pasa controles automáticos (números, nº de opciones, back-
translation) y las sospechosas van a revisión humana; (4) añade además consultas NATIVAS
(redactadas por hablantes o recogidas con licencia) para no medir solo 'inglés traducido'.
Traductor: un modelo local fuera del pool evaluado (declararlo).

## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `import json`
- `import numpy`
- `import pandas`
- `import re`

## ⚙️ Funciones globales (5)

### 📝 `_nums(s)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `s` | `Any` | — |

**Retorna:** `Any`

### 📝 `_jaccard(a, b)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `a` | `Any` | — |
| `b` | `Any` | — |

**Retorna:** `Any`

### 📝 `check_preservation(src, tgt)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `src` | `Any` | — |
| `tgt` | `Any` | — |

**Retorna:** `Any`

### 📝 `translate_queries(queries, lang, gen_fn, back_fn, min_back_jaccard)`

```python
    gen_fn(prompt)->texto (traductor). back_fn(prompt)->texto (retro-traducción, opcional).
    Devuelve (queries_traducidas, tabla_de_control con 'needs_review').
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `queries` | `Any` | — |
| `lang` | `Any` | — |
| `gen_fn` | `Any` | — |
| `back_fn` | `Any` | — |
| `min_back_jaccard` | `Any` | — |

**Retorna:** `Any`

### 📝 `review_sample(ctrl, n_random, seed)`

```python
    Todas las marcadas + muestra aleatoria del resto, para revisión humana.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `ctrl` | `Any` | — |
| `n_random` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `str`
