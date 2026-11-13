# 📄 estimate_budget.py

**Ruta relativa:** `estimate_budget.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/estimate_budget.py`

## 📖 Descripción del módulo

Estimación de GPU-horas y coste para generar datasets nuevos con pesos abiertos.
Todas las entradas son SUPUESTOS editables; calibra tok/s con una prueba corta en tu GPU.

## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `import argparse`
- `import sys`
- `import yaml`

## ⚙️ Funciones globales (2)

### 📝 `estimate(n_queries, n_cfgs, avg_in, avg_out, decode_tps, prefill_tps, n_gpus, usd_gpu_h, util)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `n_queries` | `Any` | — |
| `n_cfgs` | `Any` | — |
| `avg_in` | `Any` | — |
| `avg_out` | `Any` | — |
| `decode_tps` | `Any` | — |
| `prefill_tps` | `Any` | — |
| `n_gpus` | `Any` | — |
| `usd_gpu_h` | `Any` | — |
| `util` | `Any` | — |

**Retorna:** `Any`

### 📝 `estimate_agent_budget(n_tasks, n_models, avg_steps, avg_ctx, avg_new_per_step, avg_out, decode_tps, prefill_tps, n_gpus, usd_gpu_h, util, cache, n_cf_tasks, cf_models, cf_samples, cf_mode, cf_probe_every)`

```python
    GPU-horas de AgentRoute. Episodios fijos: n_tasks×n_models. Contrafactuales (opcional):
    local = 1 paso por (estado, modelo, muestra); rollout = pasos restantes (≈ avg_steps/2).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `n_tasks` | `Any` | — |
| `n_models` | `Any` | — |
| `avg_steps` | `Any` | — |
| `avg_ctx` | `Any` | — |
| `avg_new_per_step` | `Any` | — |
| `avg_out` | `Any` | — |
| `decode_tps` | `Any` | — |
| `prefill_tps` | `Any` | — |
| `n_gpus` | `Any` | — |
| `usd_gpu_h` | `Any` | — |
| `util` | `Any` | — |
| `cache` | `Any` | — |
| `n_cf_tasks` | `Any` | — |
| `cf_models` | `Any` | — |
| `cf_samples` | `Any` | — |
| `cf_mode` | `Any` | — |
| `cf_probe_every` | `Any` | — |

**Retorna:** `str`
