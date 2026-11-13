# 📄 pricing.py

**Ruta relativa:** `routedata/pricing.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/pricing.py`

## 📖 Descripción del módulo

Modelo de costes para modelos de pesos abiertos autoalojados.

Con GPU propia/alquilada no hay precio por token: se *deriva* del precio de la GPU y
del rendimiento medido. Documenta siempre (price_source, price_date) y repite los
análisis con escenarios de precio (±50 %) para comprobar sensibilidad.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from dataclasses import dataclass`

## 📦 Clases (1)

### 🧊 `SelfHostPrice`

---

## ⚙️ Funciones globales (5)

### 📝 `derive_selfhost_prices(gpu_usd_per_hour, n_gpus, decode_tok_per_s, prefill_tok_per_s, utilization)`

```python
    Precio efectivo por millón de tokens.

    decode_tok_per_s / prefill_tok_per_s: *throughput agregado del servidor* (todas las
    peticiones concurrentes), medido con carga realista. utilization: fracción del tiempo
    que la GPU está ocupada (penaliza el ocio).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `gpu_usd_per_hour` | `Any` | — |
| `n_gpus` | `Any` | — |
| `decode_tok_per_s` | `Any` | — |
| `prefill_tok_per_s` | `Any` | — |
| `utilization` | `Any` | — |

**Retorna:** `Any`

### 📝 `scale_prices(price, factor)`

```python
    Escenario de sensibilidad (p. ej. 0.5, 1.5).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `price` | `Any` | — |
| `factor` | `Any` | — |

**Retorna:** `Any`

### 📝 `request_cost(in_tokens, out_tokens, p)`

```python
    Coste USD de una petición; out_tokens incluye tokens de razonamiento.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `in_tokens` | `Any` | — |
| `out_tokens` | `Any` | — |
| `p` | `Any` | — |

**Retorna:** `Any`

### 📝 `throughput_from_responses(responses, configs, prefill_over_decode)`

```python
    Throughput agregado por modelo medido en un run: decode_tps = suma(out_tokens)/suma(latency_s)
    (latency_s ya es tiempo de pared del lote / nº de peticiones). El prefill NO se mide por separado:
    se asume prefill_tps = prefill_over_decode × decode_tps (ajusta con una prueba dedicada).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `responses` | `Any` | — |
| `configs` | `Any` | — |
| `prefill_over_decode` | `Any` | — |

**Retorna:** `Any`

### 📝 `reprice_configs(configs, throughput, gpu_usd_per_hour, tp_by_model, utilization, price_date)`

```python
    Rellena price_in/price_out de `configs` a partir de GPU-hora y throughput medido.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `configs` | `Any` | — |
| `throughput` | `Any` | — |
| `gpu_usd_per_hour` | `Any` | — |
| `tp_by_model` | `Any` | — |
| `utilization` | `Any` | — |
| `price_date` | `Any` | — |

**Retorna:** `Any`
