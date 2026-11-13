# 📄 base.py

**Ruta relativa:** `routedata/agents/base.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/agents/base.py`

## 📖 Descripción del módulo

Base classes for routing agents.

A routing agent takes a query (and optional candidate models) and returns
a routing decision: which model to use and why.

The agent does NOT execute the query — it only decides the route.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from dataclasses import dataclass, field`
- `from enum import Enum`
- `from typing import Any`

## 📦 Clases (4)

### 🧊 `RoutingStrategy` → str, Enum

Strategy used to make the routing decision.

**Herencia:**
```
  RoutingStrategy → str
  RoutingStrategy → Enum
```

**Atributos de clase:**

| Atributo | Descripción |
|----------|-------------|
| `COST` | — |
| `QUALITY` | — |
| `LATENCY` | — |
| `HEURISTIC` | — |
| `LEARNED` | — |

---

### 🧊 `Candidate`

A candidate model that can handle the query.

Attributes:
    name: Unique model name (e.g., "gpt-4o", "mistral-7b").
    capabilities: List of capabilities (e.g., ["text", "vision"]).
    cost_per_1k_tokens: Cost in EUR per 1k tokens (input + output average).
    quality_score: Normalized quality score [0, 1] if known, else None.
    latency_ms: Average latency in milliseconds if known, else None.
    context_window: Max context window in tokens if known, else None.
    metadata: Additional metadata.

---

### 🧊 `RoutingDecision`

The output of a routing decision.

Attributes:
    query: The original query.
    chosen_model: Name of the model to use.
    strategy: Which strategy was used.
    confidence: Confidence score [0, 1] if available.
    reasoning: Human-readable explanation of the decision.
    all_scores: Scores for all candidates (model name -> score).
    latency_ms: How long the decision took.

---

### 🧊 `RoutingAgent`

Base class for all routing agents.

Subclasses must implement:
    - decide(query, candidates) -> RoutingDecision

The agent should:
    1. Analyze the query (language, domain, complexity, etc.)
    2. Score each candidate model
    3. Select the best model
    4. Return a RoutingDecision with explanation

**Métodos:**

#### 📝 `__init__(self, config)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `config` | `Any` | — |

#### 📝 `decide(self, query, candidates)`

```python
    Make a routing decision for the given query.
    
    Args:
        query: The user's query/question.
        candidates: List of candidate models to choose from.
    
    Returns:
        A RoutingDecision with the chosen model and reasoning.
    
    Raises:
        NotImplementedError: If not implemented by subclass.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `query` | `Any` | — |
| `candidates` | `Any` | — |

#### 📝 `_log(self, message)`

```python
    Log a message (override for custom logging).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `message` | `Any` | — |

---
