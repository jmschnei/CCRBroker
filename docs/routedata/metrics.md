# 📄 metrics.py

**Ruta relativa:** `routedata/metrics.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/metrics.py`

## 📖 Descripción del módulo

Métricas de evaluación de routing sobre matrices Q (calidad) y C (coste).

Convención: Q[n,k] en [0,1] (1 = correcto), C[n,k] en USD; `choice[n]` = índice de
configuración elegida por el router para cada consulta. Las definiciones de
Gain@b, Gap@O, PerfGain y CostSave siguen LLMRouterBench; AUDC/QNC siguen la
formulación de deferral curve usada en R2-Router/UniRouter; AIQ es una
implementación propia aproximada de la de RouterBench (verificar contra el
código oficial antes de reportar números comparables).


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `import numpy`

## ⚙️ Funciones globales (21)

### 📝 `realized(Q, C, choice)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q` | `Any` | — |
| `C` | `Any` | — |
| `choice` | `Any` | — |

**Retorna:** `Any`

### 📝 `mean_quality_cost(Q, C, choice)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q` | `Any` | — |
| `C` | `Any` | — |
| `choice` | `Any` | — |

**Retorna:** `Any`

### 📝 `oracle_choice(Q, C, tau)`

```python
    Por consulta: la config más barata entre las 'correctas' (Q>=tau); si ninguna
    acierta, la más barata (no aporta calidad).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q` | `Any` | — |
| `C` | `Any` | — |
| `tau` | `Any` | — |

**Retorna:** `Any`

### 📝 `best_single(Q, groups)`

```python
    Índice del mejor modelo único *a posteriori* (media de las medias por grupo, como
    en LLMRouterBench, o media global si no hay grupos).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q` | `Any` | — |
| `groups` | `Any` | — |

**Retorna:** `Any`

### 📝 `_group_acc(q, groups)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `q` | `Any` | — |
| `groups` | `Any` | — |

**Retorna:** `Any`

### 📝 `gain_at(Q, C, choice, base_choice, groups)`

```python
    Gain@b = media_d( Acc(a,d)/Acc(b,d) - 1 ).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q` | `Any` | — |
| `C` | `Any` | — |
| `choice` | `Any` | — |
| `base_choice` | `Any` | — |
| `groups` | `Any` | — |

**Retorna:** `Any`

### 📝 `gap_at_oracle(Q, C, choice, groups, tau)`

```python
    Gap@O = media_d( 1 - Acc(a,d)/Acc(O,d) ).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q` | `Any` | — |
| `C` | `Any` | — |
| `choice` | `Any` | — |
| `groups` | `Any` | — |
| `tau` | `Any` | — |

**Retorna:** `Any`

### 📝 `recall_failure_rate(Q, choice, max_correct, tau)`

```python
    Accuracy del router en consultas 'difíciles' (<= max_correct configs aciertan).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Q` | `Any` | — |
| `choice` | `Any` | — |
| `max_correct` | `Any` | — |
| `tau` | `Any` | — |

**Retorna:** `Any`

### 📝 `pareto_frontier(points)`

```python
    points: array (m,2) con (coste, calidad). Devuelve índices no dominados
    (menor coste, mayor calidad), ordenados por coste.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `points` | `Any` | — |

**Retorna:** `Any`

### 📝 `hypervolume_2d(points, ref_cost, ref_quality)`

```python
    Área dominada respecto al punto de referencia (ref_cost alto, ref_quality bajo).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `points` | `Any` | — |
| `ref_cost` | `Any` | — |
| `ref_quality` | `Any` | — |

**Retorna:** `Any`

### 📝 `pareto_dist(method_points, frontier_points)`

```python
    Distancia L1 media a la frontera, con coordenadas normalizadas a [0,1] (ParetoDist).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `method_points` | `Any` | — |
| `frontier_points` | `Any` | — |

**Retorna:** `Any`

### 📝 `perf_gain_cost_save(points, best_single_point)`

```python
    points: [(coste, calidad)] de todas las configs de un router; best_single_point=(c_B, q_B).
    PerfGain = max_q/q_B - 1 ; CostSave = 1 - min{coste: q>=q_B}/c_B (None si no alcanza).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `points` | `Any` | — |
| `best_single_point` | `Any` | — |

**Retorna:** `Any`

### 📝 `aiq(points, c_min, c_max)`

```python
    Área normalizada bajo la envolvente cóncava no decreciente de (coste, calidad).
    Aproximación de la AIQ de RouterBench (verificar contra la implementación oficial).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `points` | `Any` | — |
| `c_min` | `Any` | — |
| `c_max` | `Any` | — |

**Retorna:** `Any`

### 📝 `deferral_curve(Qhat, Chat, Q, C, lambdas)`

```python
    Barre λ en [0,1]; el router elige argmax (1-λ)·Q̂ − λ·Ĉn (Ĉ normalizado por su máximo).
    Devuelve (coste_medio_real, calidad_media_real) por λ.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `Qhat` | `Any` | — |
| `Chat` | `Any` | — |
| `Q` | `Any` | — |
| `C` | `Any` | — |
| `lambdas` | `Any` | — |

**Retorna:** `Any`

### 📝 `audc(curve, cost_norm)`

```python
    Área bajo la deferral curve con coste normalizado por `cost_norm`
    (p. ej. coste medio de la config más cara). La curva se hace monótona.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `curve` | `Any` | — |
| `cost_norm` | `Any` | — |

**Retorna:** `Any`

### 📝 `qnc(curve, best_model_quality, best_model_cost)`

```python
    Query-Normalized Cost: coste mínimo relativo para igualar la calidad del mejor
    modelo (1.0 si no se alcanza).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `curve` | `Any` | — |
| `best_model_quality` | `Any` | — |
| `best_model_cost` | `Any` | — |

**Retorna:** `Any`

### 📝 `consistency(choice_a, choice_b)`

```python
    Fracción de consultas donde el router elige lo mismo ante la consulta original y
    la perturbada (consistencia ante perturbaciones).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `choice_a` | `Any` | — |
| `choice_b` | `Any` | — |

**Retorna:** `Any`

### 📝 `cost_inflation(C, choice_clean, choice_attacked)`

```python
    Multiplicador de coste medio bajo ataque.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `C` | `Any` | — |
| `choice_clean` | `Any` | — |
| `choice_attacked` | `Any` | — |

**Retorna:** `Any`

### 📝 `ece(p, y, bins)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `p` | `Any` | — |
| `y` | `Any` | — |
| `bins` | `Any` | — |

**Retorna:** `Any`

### 📝 `brier(p, y)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `p` | `Any` | — |
| `y` | `Any` | — |

**Retorna:** `Any`

### 📝 `risk_coverage(conf, y)`

```python
    Curva riesgo-cobertura: ordena por confianza descendente.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `conf` | `Any` | — |
| `y` | `Any` | — |

**Retorna:** `Any`
