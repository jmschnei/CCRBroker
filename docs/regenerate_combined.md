# 📄 regenerate_combined.py

**Ruta relativa:** `regenerate_combined.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/regenerate_combined.py`

## 📖 Descripción del módulo

Regenera los archivos unificados (por dataset + combinados + manifest).

Solo carga los 3 datasets integrados actualmente. R2-Bench queda excluido
por tamaño (ver PIPELINE_SUMMARY.md); para añadirlo, ve el adaptador
`routedata/adapters/r2bench.py`.

Reutiliza `Pipeline.save` de `routedata.pipeline` para garantizar el esquema
y el etiquetado `source_dataset` consistentes.

Uso:  python regenerate_combined.py


## 🔗 Dependencias (imports)

- `from pathlib import Path`
- `from routedata.adapters.llmrouterbench import LLMRouterBenchAdapter`
- `from routedata.adapters.routerbench import load_both`
- `from routedata.adapters.sprout import create_sprout_adapter`
- `from routedata.pipeline import Pipeline`
- `import pandas`

## ⚙️ Funciones globales (2)

### 📝 `load_results()`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|

**Retorna:** `Any`

### 📝 `main()`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|

**Retorna:** `Any`
