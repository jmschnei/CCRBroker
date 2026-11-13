# 📄 adapters_template.py

**Ruta relativa:** `generate/envs/adapters_template.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/envs/adapters_template.py`

## 📖 Descripción del módulo

PLANTILLAS de adaptadores a entornos agénticos reales. NO FUNCIONAN TAL CUAL: no he podido
verificar las APIs de estos paquetes desde este entorno (sin red). Cada clase indica qué implementar.

Contrato (ver `generate/agentroute.py::Env`):
  reset(seed) -> str         observación inicial (instrucciones + estado)
  step(action:str) -> (obs, done)   el texto del modelo es la acción; tú lo parseas (JSON de herramienta, ReAct...)
                                    errores de herramienta/formato => obs que EMPIECE por "ERROR"
  success() -> float         éxito de la tarea
  clone() -> Env             ReplayCloneMixin sirve si el entorno es determinista dada (semilla, acciones)
Atributos opcionales que usa el MockBackend: gold_action, difficulty.
Recomendación de orden por viabilidad (sin verificar): ScienceWorld (texto, sin simulador de usuario) >
QA multi-salto con herramienta de búsqueda local > τ²-Bench (simulador de usuario LLM: difícil de hacer
determinista) > SWE-bench Verified (contenedores, trayectorias largas y caras).


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from agentroute import ReplayCloneMixin`

## 📦 Clases (4)

### 🧊 `ScienceWorldEnv` → ReplayCloneMixin

TODO: envolver el paquete `scienceworld` (ScienceWorldEnv): load(task, variation), reset(), step(action).
Mapear `seed` -> (tarea, variación); success() -> puntuación normalizada; comprobar determinismo.

**Herencia:**
```
  ScienceWorldEnv → ReplayCloneMixin
```

**Métodos:**

#### 📝 `_make_fresh(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

---

### 🧊 `Tau2Env` → ReplayCloneMixin

TODO: envolver τ²-Bench. Necesita un simulador de usuario (usa un modelo LOCAL con T=0 y semilla fija) y
un parser de llamadas a herramientas. Verifica que dos ejecuciones con las mismas acciones den las mismas
observaciones (`check_deterministic`); si no, los contrafactuales no son válidos.

**Herencia:**
```
  Tau2Env → ReplayCloneMixin
```

**Métodos:**

#### 📝 `_make_fresh(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

---

### 🧊 `SWEBenchEnv` → ReplayCloneMixin

TODO: contenedor por tarea; success() = tests de la tarea. Cada rollout contrafactual relanza el historial:
muy caro. Limita a un subconjunto pequeño o prescinde de contrafactuales aquí.

**Herencia:**
```
  SWEBenchEnv → ReplayCloneMixin
```

**Métodos:**

#### 📝 `_make_fresh(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

---

### 🧊 `MultiHopQAEnv` → ReplayCloneMixin

TODO: herramienta `search(query)` sobre un corpus local (BM25 o embeddings fijos) + `answer(text)`.
Determinista por construcción: la opción más barata para empezar.

**Herencia:**
```
  MultiHopQAEnv → ReplayCloneMixin
```

**Métodos:**

#### 📝 `_make_fresh(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

---
