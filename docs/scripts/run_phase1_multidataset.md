# 📄 run_phase1_multidataset.py

**Ruta relativa:** `scripts/run_phase1_multidataset.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/scripts/run_phase1_multidataset.py`

## 📖 Descripción del módulo

Audit and run the LLMRouterBench performance-only 15-dataset protocol.

Protocol follows this repository's PLAN: the official 20-model lightweight pool,
70/30 train/test split, five declared seeds, and per-dataset splits grouped by
identical prompt so all model outcomes stay on the same side. The upstream
LLMRouterBench loader documents this dataset-then-prompt grouping, while its
current generic defaults are 80/20 and seed 42; this script applies the local
PLAN's explicit 70/30 and five seeds rather than claiming a byte-identical
upstream split implementation.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from collections import defaultdict`
- `from pathlib import Path`
- `from routedata.splits import SEEDS_LLMROUTERBENCH`
- `from sklearn.feature_extraction.text import TfidfVectorizer`
- `from sklearn.linear_model import LogisticRegression, Ridge`
- `from sklearn.metrics.pairwise import cosine_similarity`
- `from typing import Any`
- `import argparse`
- `import json`
- `import numpy`
- `import pandas`
- `import random`
- `import re`
- `import sys`

## ⚙️ Funciones globales (10)

### 📝 `_index_from_query_id(query_id)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `query_id` | `Any` | — |

**Retorna:** `Any`

### 📝 `split_dataset_by_prompt(query_ids, texts, train_ratio, seed, rng)`

```python
    Split one dataset by unique prompt, mirroring upstream leakage prevention.

    Prompt groups are ordered by the minimum numeric source index; a Python
    Random instance shuffles prompt positions, then the first floor(N*ratio)
    groups are assigned to train. All rows sharing prompt text remain together.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `query_ids` | `Any` | — |
| `texts` | `Any` | — |
| `train_ratio` | `Any` | — |
| `seed` | `Any` | — |
| `rng` | `Any` | — |

**Retorna:** `str`

### 📝 `audit_data(unified_dir)`

```python
    Audit local pool, expected tasks, labels, score variation, and matrix coverage.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `unified_dir` | `Any` | — |

**Retorna:** `Any`

### 📝 `_argmax_stable(values, model_ids)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `values` | `Any` | — |
| `model_ids` | `Any` | — |

**Retorna:** `Any`

### 📝 `_fit_predict(strategy, train_texts, test_texts, train_scores, model_ids, seed)`

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

### 📝 `_summarize_seed(dataset_results, dataset_sizes, strategies)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `dataset_results` | `Any` | — |
| `dataset_sizes` | `Any` | — |
| `strategies` | `Any` | — |

**Retorna:** `Any`

### 📝 `run_evaluation(experiment_data, train_ratio, seeds)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `experiment_data` | `Any` | — |
| `train_ratio` | `Any` | — |
| `seeds` | `Any` | — |

**Retorna:** `Any`

### 📝 `_json_safe(value)`

```python
    Convert NumPy scalars and non-finite floats to strict JSON-compatible values.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `value` | `Any` | — |

**Retorna:** `str`

### 📝 `main(argv)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `argv` | `Any` | — |

**Retorna:** `Any`
