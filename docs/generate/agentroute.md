# 📄 agentroute.py

**Ruta relativa:** `generate/agentroute.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/agentroute.py`

## 📖 Descripción del módulo

AgentRoute (Fase 4): trayectorias multi-turno, políticas mixtas, coste con caché de prefijos y
utilidades contrafactuales por paso.

Contabilidad de coste (clave para H5). En cada paso el servidor recibe TODO el contexto. Dos modos:
  cache_mode="none"   : cada paso factura el contexto completo (el cambio de modelo no cuesta extra).
  cache_mode="prefix" : con caché automática de prefijos solo se factura lo no cacheado; la caché es
                        PROPIA DE CADA MODELO, así que al cambiar de modelo hay que re-procesar el
                        contexto en el modelo destino (salvo lo que conserve de llamadas previas si
                        retention="keep").
El coste se RECALCULA offline desde el log (tokens de contexto, de salida y modelo por paso): puedes
cambiar precios, modo de caché y retención sin volver a ejecutar episodios.

Entornos reales (τ²-Bench, ScienceWorld, SWE-bench...): implementa el protocolo `Env` en
`generate/envs/` (ver plantillas). Los contrafactuales requieren `Env.clone()`; con simuladores
externos usa `ReplayCloneMixin` (reproduce el historial de acciones) y comprueba determinismo.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from backends import GenRequest`
- `from dataclasses import dataclass, asdict, field`
- `from typing import Callable, Protocol`
- `import copy`
- `import numpy`

## 📦 Clases (6)

### 🧊 `Env` → Protocol

**Herencia:**
```
  Env → Protocol
```

**Métodos:**

#### 📝 `reset(self, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `seed` | `Any` | — |

#### 📝 `step(self, action)`

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

#### 📝 `clone(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

---

### 🧊 `StepLog`

---

### 🧊 `Episode`

**Métodos:**

#### 📝 `totals(self, price, cache_mode, retention)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `price` | `Any` | — |
| `cache_mode` | `Any` | — |
| `retention` | `Any` | — |

---

### 🧊 `AgentState`

**Métodos:**

#### 📝 `__init__(self, env, backends, seed, system, episode_id)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `env` | `Any` | — |
| `backends` | `Any` | — |
| `seed` | `Any` | — |
| `system` | `Any` | — |
| `episode_id` | `Any` | — |

#### 📝 `step(self, model, max_tokens)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `model` | `Any` | — |
| `max_tokens` | `Any` | — |

#### 📝 `clone(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

---

### 🧊 `ReplayCloneMixin`

clone() reproduciendo el historial de acciones en un entorno nuevo con la misma semilla.
Requiere un entorno DETERMINISTA dado (semilla, acciones). Si el entorno incluye un simulador de
usuario con LLM, fija temperatura 0 y semilla, y verifica con `check_deterministic`.

**Métodos:**

#### 📝 `_make_fresh(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

#### 📝 `clone(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

---

### 🧊 `ToyEnv`

Entorno mínimo (compatibilidad con pruebas anteriores).

**Métodos:**

#### 📝 `__init__(self, horizon, difficulty)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `horizon` | `Any` | — |
| `difficulty` | `Any` | — |

#### 📝 `reset(self, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |
| `seed` | `Any` | — |

#### 📝 `step(self, action)`

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

#### 📝 `clone(self)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `self` | `Any` | — |

---

## ⚙️ Funciones globales (15)

### 📝 `_pp(p)`

```python
    Precio de un modelo: tupla (in,out) o dict {'in','out','prefill_tps','decode_tps'} en USD/Mtok.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `p` | `Any` | — |

**Retorna:** `Any`

### 📝 `billed_prefill(steps, cache_mode, retention)`

```python
    Tokens de prefill facturados por paso. retention='keep': la caché de cada modelo sobrevive a las
    llamadas de otros modelos; 'evict': solo cuenta si el paso anterior fue del mismo modelo.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `steps` | `Any` | — |
| `cache_mode` | `Any` | — |
| `retention` | `Any` | — |

**Retorna:** `Any`

### 📝 `candidate_prefill(history, model, ctx_next, cache_mode, retention)`

```python
    Prefill que se facturaría si el próximo paso lo hiciera `model` con contexto `ctx_next`.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `history` | `Any` | — |
| `model` | `Any` | — |
| `ctx_next` | `Any` | — |
| `cache_mode` | `Any` | — |
| `retention` | `Any` | — |

**Retorna:** `Any`

### 📝 `steps_cost(steps, price, cache_mode, retention, start)`

```python
    Coste USD (y latencia estimada si hay tps) de los pasos desde `start`.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `steps` | `Any` | — |
| `price` | `Any` | — |
| `cache_mode` | `Any` | — |
| `retention` | `Any` | — |
| `start` | `Any` | — |

**Retorna:** `Any`

### 📝 `fixed_policy(model)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `model` | `Any` | — |

**Retorna:** `Any`

### 📝 `switch_every_k(models, k)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `models` | `Any` | — |
| `k` | `Any` | — |

**Retorna:** `Any`

### 📝 `random_policy(models, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `models` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `escalation_policy(cheap, strong, k_fail)`

```python
    Baseline práctico: empieza barato y escala (sin volver) tras k errores de herramienta consecutivos.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `cheap` | `Any` | — |
| `strong` | `Any` | — |
| `k_fail` | `Any` | — |

**Retorna:** `Any`

### 📝 `approx_context(history, obs)`

```python
    Estimación del contexto del próximo paso (para decidir antes de llamar al modelo).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `history` | `Any` | — |
| `obs` | `Any` | — |

**Retorna:** `Any`

### 📝 `run_episode(env, backends, policy, episode_id, policy_name, seed, max_steps, system, task_id, env_name)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `env` | `Any` | — |
| `backends` | `Any` | — |
| `policy` | `Any` | — |
| `episode_id` | `Any` | — |
| `policy_name` | `Any` | — |
| `seed` | `Any` | — |
| `max_steps` | `Any` | — |
| `system` | `Any` | — |
| `task_id` | `Any` | — |
| `env_name` | `Any` | — |

**Retorna:** `Any`

### 📝 `rollout_from(state, first_model, continue_model, max_steps, seed_off)`

```python
    Desde un estado, elige `first_model` ahora y continúa con `continue_model` hasta el final.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `state` | `Any` | — |
| `first_model` | `Any` | — |
| `continue_model` | `Any` | — |
| `max_steps` | `Any` | — |
| `seed_off` | `Any` | — |

**Retorna:** `Any`

### 📝 `epsilon_policy(ref, models, eps, seed)`

```python
    Política de comportamiento para recoger estados: el modelo de referencia con prob. 1-eps, aleatorio con eps.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `ref` | `Any` | — |
| `models` | `Any` | — |
| `eps` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `probe_step(state, model, seed_off)`

```python
    Muestra UN paso con `model` desde el estado y devuelve (acción válida?, tokens de salida). Etiqueta LOCAL barata:
    requiere que el entorno señale errores de herramienta/formato con obs que empieza por 'ERROR'.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `state` | `Any` | — |
| `model` | `Any` | — |
| `seed_off` | `Any` | — |

**Retorna:** `str`

### 📝 `collect_step_utilities(env_factory, task_seed, backends, models, behavior_policy, reference_model, price, n_rollouts, max_steps, probe_every, cache_mode, retention, task_id, mode)`

```python
    Para cada estado visitado por la política de comportamiento y cada modelo candidato m: usar m AHORA y continuar
    con `reference_model`. Por fila: utilidad (éxito medio de n_rollouts), tokens de salida del primer paso y coste
    esperado POSTERIOR al primer paso (`cost_after`, USD). El coste del primer paso NO se aprende: se calcula exacto en
    la decisión con el estado real de la caché (así el coste de cambiar de modelo es exacto).
    mode="local" (por defecto): utilidad = P(acción válida en este paso) con n muestras de UN solo paso (barato, sin
    continuación; cost_after=0). mode="rollout": continúa con el modelo de referencia hasta el final (caro, captura
    consecuencias a largo plazo; tenga en cuenta que el coste posterior incluye el cambio al modelo de referencia).
    COSTE rollout ≈ pasos_sondeados × n_modelos × n_rollouts × pasos_restantes (ver estimate_budget.py --agent-cf).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `env_factory` | `Any` | — |
| `task_seed` | `Any` | — |
| `backends` | `Any` | — |
| `models` | `Any` | — |
| `behavior_policy` | `Any` | — |
| `reference_model` | `Any` | — |
| `price` | `Any` | — |
| `n_rollouts` | `Any` | — |
| `max_steps` | `Any` | — |
| `probe_every` | `Any` | — |
| `cache_mode` | `Any` | — |
| `retention` | `Any` | — |
| `task_id` | `Any` | — |
| `mode` | `Any` | — |

**Retorna:** `str`

### 📝 `check_deterministic(env_factory, seed, actions, n)`

```python
    Fracción de ejecuciones cuyas observaciones coinciden con la primera. Debe ser ≈ 1 antes de
    fiarte de los contrafactuales.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `env_factory` | `Any` | — |
| `seed` | `Any` | — |
| `actions` | `Any` | — |
| `n` | `Any` | — |

**Retorna:** `Any`
