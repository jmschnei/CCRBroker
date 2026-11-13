# 📄 backends.py

**Ruta relativa:** `generate/backends.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/backends.py`

## 📖 Descripción del módulo

Backends de generación. Producción: vLLM con pesos abiertos (pin de revisión).
Pruebas: MockBackend determinista (sin GPU).

## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from dataclasses import dataclass, field`
- `from typing import Protocol`
- `from vllm import LLM`
- `from vllm import SamplingParams`
- `import hashlib`
- `import json`
- `import time`
- `import urllib.request`

## 📦 Clases (6)

### 🧊 `GenRequest`

---

### 🧊 `GenResult`

---

### 🧊 `Backend` → Protocol

**Herencia:**
```
  Backend → Protocol
```

**Métodos:**

#### 📝 `generate(self, reqs)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `reqs` | `Any` | — |

---

### 🧊 `VLLMBackend`

Requiere `pip install vllm` y GPU. `revision` = commit de HF (OBLIGATORIO para
reproducibilidad y para DriftRoute).

**Métodos:**

#### 📝 `__init__(self, model, revision, tensor_parallel_size, dtype, max_model_len, **kw)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `model` | `Any` | — |
| `revision` | `Any` | — |
| `tensor_parallel_size` | `Any` | — |
| `dtype` | `Any` | — |
| `max_model_len` | `Any` | — |
| `**kw` | `Any` | — |

#### 📝 `_prompt(self, r)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `r` | `Any` | — |

#### 📝 `generate(self, reqs)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `reqs` | `Any` | — |

---

### 🧊 `MockBackend`

Simulador determinista: la probabilidad de acierto depende de la habilidad del modelo,
la dificultad de la consulta (meta['difficulty'] ∈ [0,1]), el presupuesto de tokens y
el esfuerzo. Devuelve la respuesta correcta (meta['answer']) o una incorrecta.

**Métodos:**

#### 📝 `__init__(self, name, skill, verbosity)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `name` | `Any` | — |
| `skill` | `Any` | — |
| `verbosity` | `Any` | — |

#### 📝 `_u(self, r, tag)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `r` | `Any` | — |
| `tag` | `Any` | — |

#### 📝 `generate(self, reqs)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `reqs` | `Any` | — |

---

### 🧊 `OpenAICompatBackend`

Cliente para servidores compatibles con OpenAI (p. ej. `vllm serve`). Permite episodios
CONCURRENTES (hilos) contra un servidor con batching continuo, que es lo que da el throughput
por el que se deriva el precio por token. Solo usa la biblioteca estándar.

NO verificado contra un servidor real desde este entorno (probado con un servidor falso local).
`cached_tokens` se lee de usage.prompt_tokens_details si el servidor lo expone (depende de la versión).

**Métodos:**

#### 📝 `__init__(self, base_url, model, api_key, timeout)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `base_url` | `Any` | — |
| `model` | `Any` | — |
| `api_key` | `Any` | — |
| `timeout` | `Any` | — |

#### 📝 `_one(self, r)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `r` | `Any` | — |

#### 📝 `generate(self, reqs)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `reqs` | `Any` | — |

---
