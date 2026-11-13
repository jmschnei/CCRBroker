# 📄 benchmark_vllm.py

**Ruta relativa:** `scripts/benchmark_vllm.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/scripts/benchmark_vllm.py`

## 📖 Descripción del módulo

Measure aggregate prefill/decode throughput for one pinned pool model on CUDA.

Run on a Linux host with NVIDIA CUDA and the optional vLLM dependency installed.
The script does not download data or infer GPU pricing; pass a verified hourly rate.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from datetime import datetime, timezone`
- `from generate.backends import GenRequest, VLLMBackend`
- `from pathlib import Path`
- `from routedata.pricing import derive_selfhost_prices, scale_prices`
- `from typing import Any`
- `import argparse`
- `import importlib.metadata`
- `import json`
- `import platform`
- `import sys`
- `import time`
- `import torch`
- `import yaml`

## ⚙️ Funciones globales (3)

### 📝 `_build_prompt(tokenizer, target_tokens)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `tokenizer` | `Any` | — |
| `target_tokens` | `Any` | — |

**Retorna:** `Any`

### 📝 `_batches(items, batch_size)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `items` | `Any` | — |
| `batch_size` | `Any` | — |

**Retorna:** `Any`

### 📝 `main(argv)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `argv` | `Any` | — |

**Retorna:** `Any`
