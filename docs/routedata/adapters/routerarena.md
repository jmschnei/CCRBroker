# 📄 routerarena.py

**Ruta relativa:** `routedata/adapters/routerarena.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/adapters/routerarena.py`

## 📖 Descripción del módulo

RouterArena (RouteWorks/RouterArena en HF): conjunto de SOLO EVALUACIÓN.

Hechos verificados: 8 400 preguntas (split `full`), 809 (`sub_10`) y 420 (`robustness`);
9 dominios, 44 categorías; niveles de dificultad fácil/medio/difícil. El README del
proyecto prohíbe entrenar o ajustar componentes del router con estos datos (incluidas las
etiquetas) para enviar al leaderboard. Este adaptador marca `eval_only=True` y
`assert_trainable` impide usarlo para entrenar.
Los esquemas de columnas han variado entre revisiones del dataset en HF: verifica.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from datasets import load_dataset`
- `import pandas`

## ⚙️ Funciones globales (2)

### 📝 `to_queries(df)`

```python
    df con columnas 'Question','Context','Options','Answer','Domain','Category',
    'Difficulty','Global Index' (si existen).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `Any` | — |

**Retorna:** `Any`

### 📝 `load_hf(split)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `split` | `Any` | — |

**Retorna:** `Any`
