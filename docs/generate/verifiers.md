# 📄 verifiers.py

**Ruta relativa:** `generate/verifiers.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/verifiers.py`

## 📖 Descripción del módulo

Verificadores programáticos (etiqueta 'correct' ∈ {0,1}). Para tareas abiertas usa judge.py.

## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `import os`
- `import re`
- `import subprocess`
- `import sys`
- `import tempfile`

## ⚙️ Funciones globales (8)

### 📝 `extract_final(text)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `Any` | — |

**Retorna:** `Any`

### 📝 `extract_letter(text)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `Any` | — |

**Retorna:** `Any`

### 📝 `_num(s)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `s` | `Any` | — |

**Retorna:** `Any`

### 📝 `verify_numeric(text, gold, rel_tol)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `Any` | — |
| `gold` | `Any` | — |
| `rel_tol` | `Any` | — |

**Retorna:** `Any`

### 📝 `verify_mc(text, gold_letter)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `Any` | — |
| `gold_letter` | `Any` | — |

**Retorna:** `Any`

### 📝 `verify_exact(text, gold)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `Any` | — |
| `gold` | `Any` | — |

**Retorna:** `Any`

### 📝 `run_code_tests(code, tests, timeout)`

```python
    Ejecuta código generado + tests en un subproceso. ATENCIÓN: no es un sandbox
    completo; ejecútalo SOLO en contenedor/VM aislados, sin red y sin credenciales.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `code` | `Any` | — |
| `tests` | `Any` | — |
| `timeout` | `Any` | — |

**Retorna:** `Any`

### 📝 `verify(ttype, text, gold, tests)`

```python
    Devuelve (correct, label_source) o (None, None) si requiere juez.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `ttype` | `Any` | — |
| `text` | `Any` | — |
| `gold` | `Any` | — |
| `tests` | `Any` | — |

**Retorna:** `Any`
