# 📄 routerbench.py

**Ruta relativa:** `routedata/adapters/routerbench.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/adapters/routerbench.py`

## 📖 Descripción del módulo

RouterBench (withmartian/routerbench en Hugging Face).

Hechos verificados (ficha HF): ~30 000 prompts y respuestas de 11 LLM; prompts de MBPP,
GSM8K, Winogrande, HellaSwag, MMLU, MT-Bench, etc.; hay versión 0-shot y 5-shot; los
ficheros son pickles (p. ej. routerbench_raw.pkl).
NO verificado: nombres exactos de columnas. Se asume (por el paquete oficial) que hay una
columna de prompt, una de nombre de evaluación y, por modelo, una columna de calidad y otra
'<modelo>|total_cost'. Ejecuta `python -m routedata.adapters.routerbench inspect <pkl>` primero.

SEGURIDAD: un pickle puede ejecutar código al cargarse. Verifica el hash/origen y cárgalo
en un entorno aislado.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from base import wide_to_tables`
- `import pandas`
- `import sys`

## ⚙️ Funciones globales (1)

### 📝 `load(path, prompt_col, domain_col, id_col, source_name)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `path` | `Any` | — |
| `prompt_col` | `Any` | — |
| `domain_col` | `Any` | — |
| `id_col` | `Any` | — |
| `source_name` | `Any` | — |

**Retorna:** `Any`
