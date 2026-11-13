# 📄 prompts.py

**Ruta relativa:** `generate/prompts.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/prompts.py`

## 📖 Descripción del módulo

Plantillas de prompt y restricciones de longitud ("use at most k tokens", como R2-Router).

## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `import json`

## ⚙️ Funciones globales (3)

### 📝 `length_instruction(k)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `k` | `Any` | — |

**Retorna:** `Any`

### 📝 `build_messages(row, budget, system)`

```python
    row: fila de `queries`. El tipo de tarea se deduce de `options`/`meta['task_type']`.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `row` | `Any` | — |
| `budget` | `Any` | — |
| `system` | `Any` | — |

**Retorna:** `Any`

### 📝 `compliance(out_tokens, budget, slack)`

```python
    Cumplimiento de la restricción de longitud (R2-Bench: longitud ≤ 1.1×presupuesto).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `out_tokens` | `Any` | — |
| `budget` | `Any` | — |
| `slack` | `Any` | — |

**Retorna:** `str`
