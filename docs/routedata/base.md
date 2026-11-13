# 📄 base.py

**Ruta relativa:** `routedata/base.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/base.py`

## 📖 Descripción del módulo

Adaptadores: convierten datasets de routing públicos al esquema unificado.

IMPORTANTE: los esquemas de archivos externos no se han podido inspeccionar desde este
entorno (sin red). Los adaptadores son *defensivos* (autodetección + mapeo por YAML) y
cada uno trae un comando `inspect` que imprime columnas para que verifiques el mapeo
antes de usarlo.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `import pandas`

## 📦 Clases (1)

### 🧊 `EvalOnlyError` → RuntimeError

**Herencia:**
```
  EvalOnlyError → RuntimeError
```

---

## ⚙️ Funciones globales (2)

### 📝 `assert_trainable(queries)`

```python
    Falla si alguna consulta pertenece a un dataset de solo evaluación (p. ej. RouterArena).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `queries` | `Any` | — |

**Retorna:** `Any`

### 📝 `wide_to_tables(df, prompt_col, perf_cols, cost_suffix, cost_cols, domain_col, id_col, source_name, eval_only, answer_col)`

```python
    Formato ancho (una fila por consulta; columnas por modelo) → (queries, configs, responses).
    
    perf_cols: columnas con la calidad por modelo. El coste por modelo se toma de
    cost_cols[modelo] o de f"{modelo}{cost_suffix}". Como solo hay coste final (no tokens),
    los precios por token quedan en 0 y se usa `cost_usd`.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `Any` | — |
| `prompt_col` | `Any` | — |
| `perf_cols` | `Any` | — |
| `cost_suffix` | `Any` | — |
| `cost_cols` | `Any` | — |
| `domain_col` | `Any` | — |
| `id_col` | `Any` | — |
| `source_name` | `Any` | — |
| `eval_only` | `Any` | — |
| `answer_col` | `Any` | — |

**Retorna:** `Any`
