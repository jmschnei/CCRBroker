# 📄 baseline.py

**Ruta relativa:** `routedata/agents/baseline.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/agents/baseline.py`

## 📖 Descripción del módulo

Baseline routing agents for testing and comparison.

These agents use simple, explainable strategies:

1. **RandomAgent**: Randomly picks a model (baseline).
2. **CostAgent**: Always picks the cheapest model.
3. **QualityAgent**: Always picks the highest-quality model.
4. **LatencyAgent**: Always picks the lowest-latency model.
5. **HybridAgent**: Combines cost + quality + latency with weights.
6. **HeuristicAgent**: Uses simple heuristics based on query features
   (length, domain, language, complexity).


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from base import RoutingAgent, RoutingDecision, Candidate, RoutingStrategy`
- `from typing import Any`
- `import hashlib`
- `import random`
- `import re`

## 📦 Clases (6)

### 🧊 `RandomAgent` → RoutingAgent

Baseline: randomly picks a model.

**Herencia:**
```
  RandomAgent → RoutingAgent
```

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `name` | — |

**Métodos:**

#### 📝 `__init__(self, config, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `config` | `Any` | — |
| `seed` | `Any` | — |

#### 📝 `decide(self, query, candidates)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `query` | `Any` | — |
| `candidates` | `Any` | — |

---

### 🧊 `CostAgent` → RoutingAgent

Baseline: always picks the cheapest model.

**Herencia:**
```
  CostAgent → RoutingAgent
```

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `name` | — |

**Métodos:**

#### 📝 `decide(self, query, candidates)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `query` | `Any` | — |
| `candidates` | `Any` | — |

---

### 🧊 `QualityAgent` → RoutingAgent

Baseline: always picks the highest-quality model.

**Herencia:**
```
  QualityAgent → RoutingAgent
```

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `name` | — |

**Métodos:**

#### 📝 `decide(self, query, candidates)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `query` | `Any` | — |
| `candidates` | `Any` | — |

---

### 🧊 `LatencyAgent` → RoutingAgent

Baseline: always picks the lowest-latency model.

**Herencia:**
```
  LatencyAgent → RoutingAgent
```

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `name` | — |

**Métodos:**

#### 📝 `decide(self, query, candidates)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `query` | `Any` | — |
| `candidates` | `Any` | — |

---

### 🧊 `HybridAgent` → RoutingAgent

Combines cost, quality, and latency with configurable weights.

Score = w_quality * quality - w_cost * cost - w_latency * latency

**Herencia:**
```
  HybridAgent → RoutingAgent
```

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `name` | — |

**Métodos:**

#### 📝 `__init__(self, config)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `config` | `Any` | — |

#### 📝 `decide(self, query, candidates)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `query` | `Any` | — |
| `candidates` | `Any` | — |

---

### 🧊 `HeuristicAgent` → RoutingAgent

Uses simple heuristics based on query features.

Features extracted:
- Language (English, Spanish, German, etc.)
- Domain (legal, medical, technical, general)
- Complexity (number of words, presence of technical terms)
- Query type (factual, analytical, creative)

**Herencia:**
```
  HeuristicAgent → RoutingAgent
```

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `name` | — |
| `DOMAIN_KEYWORDS` | — |

**Métodos:**

#### 📝 `_detect_language(self, text)`

```python
    Simple language detection based on common words.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `text` | `Any` | — |

#### 📝 `_detect_domain(self, text)`

```python
    Detect domain based on keywords.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `text` | `Any` | — |

#### 📝 `_estimate_complexity(self, text)`

```python
    Estimate complexity based on word count and technical terms.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `text` | `Any` | — |

#### 📝 `_select_model_by_domain(self, domain, candidates)`

```python
    Select a model based on domain (if candidates have domain-specific scores).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `domain` | `Any` | — |
| `candidates` | `Any` | — |

#### 📝 `decide(self, query, candidates)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `query` | `Any` | — |
| `candidates` | `Any` | — |

---

## ⚙️ Funciones globales (1)

### 📝 `create_agent(name, config)`

```python
    Create a routing agent by name.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `name` | `Any` | — |
| `config` | `Any` | — |

**Retorna:** `Any`
