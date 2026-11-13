# 📄 run_phase1.py

**Ruta relativa:** `scripts/run_phase1.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/scripts/run_phase1.py`

## 📖 Descripción del módulo

Run the five-seed LLMRouterBench Phase 1 model-selection experiment.

This implementation deliberately uses only ArenaHard's per-instance labels and
an explicit 20-model open-weight candidate list. It does not treat aggregate
benchmark scores as per-query labels and does not report costs/latency where the
source data has no meaningful cost variation or timing observations.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from pathlib import Path`
- `from routedata.splits import SEEDS_LLMROUTERBENCH, iid_split`
- `from sklearn.feature_extraction.text import TfidfVectorizer`
- `from sklearn.linear_model import LogisticRegression`
- `from sklearn.metrics.pairwise import cosine_similarity`
- `from typing import Any`
- `import argparse`
- `import json`
- `import numpy`
- `import pandas`
- `import sys`

## ⚙️ Funciones globales (6)

### 📝 `load_experiment_data(unified_dir, dataset)`

```python
    Return sorted query texts, fixed model IDs, and a complete per-instance score matrix.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `unified_dir` | `Any` | — |
| `dataset` | `Any` | — |

**Retorna:** `Any`

### 📝 `_argmax_stable(values, model_ids)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `values` | `Any` | — |
| `model_ids` | `Any` | — |

**Retorna:** `Any`

### 📝 `_fit_predict(strategy, train_texts, test_texts, train_scores, model_ids, seed)`

```python
    Fit a routing baseline on train outcomes only and return one model index per test query.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `strategy` | `Any` | — |
| `train_texts` | `Any` | — |
| `test_texts` | `Any` | — |
| `train_scores` | `Any` | — |
| `model_ids` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `_score_choices(test_scores, choices)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `test_scores` | `Any` | — |
| `choices` | `Any` | — |

**Retorna:** `Any`

### 📝 `run_experiment(unified_dir, dataset)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `unified_dir` | `Any` | — |
| `dataset` | `Any` | — |

**Retorna:** `Any`

### 📝 `main(argv)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `argv` | `Any` | — |

**Retorna:** `Any`
