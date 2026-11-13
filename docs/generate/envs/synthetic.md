# 📄 synthetic.py

**Ruta relativa:** `generate/envs/synthetic.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/envs/synthetic.py`

## 📖 Descripción del módulo

Entorno sintético para ensayos en seco: pasos fáciles y difíciles con observaciones largas
(el contexto crece). Un paso fallido se reintenta; el éxito exige completar todos los pasos dentro del tope de pasos.
Con MockBackend, un modelo barato falla los pasos difíciles y uno fuerte los acierta: el enrutado
por paso tiene margen, y el cambio de modelo cuesta prefill cuando hay caché de prefijos.

## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from agentroute import ReplayCloneMixin`
- `import numpy`

## 📦 Clases (1)

### 🧊 `SyntheticAgentEnv` → ReplayCloneMixin

**Herencia:**
```
  SyntheticAgentEnv → ReplayCloneMixin
```

**Métodos:**

#### 📝 `__init__(self, horizon, n_hard, obs_chars, hard_d, easy_d)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `horizon` | `Any` | — |
| `n_hard` | `Any` | — |
| `obs_chars` | `Any` | — |
| `hard_d` | `Any` | — |
| `easy_d` | `Any` | — |

#### 📝 `_make_fresh(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

#### 📝 `_obs(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

#### 📝 `reset(self, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `seed` | `Any` | — |

#### 📝 `step(self, action)`

```python
    Una acción válida ('ok') avanza el paso; una inválida devuelve ERROR y obliga a REINTENTAR el mismo paso
    (cuesta pasos y contexto, y el episodio tiene un tope de pasos en el ejecutor).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `action` | `Any` | — |

#### 📝 `success(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

---
