# 📄 advroute.py

**Ruta relativa:** `generate/advroute.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/advroute.py`

## 📖 Descripción del módulo

AdvRoute: perturbaciones benignas y ataques de rerouting (evaluación DEFENSIVA).

Modelo de amenaza (declararlo en el artículo): el atacante controla solo el texto de su
consulta (sin acceso al router en black-box; con logits/embeddings en white-box), y su objetivo
es (a) inflar el coste forzando el modelo caro o (b) degradar la calidad forzando el barato.
Referencias: Shafran et al. (arXiv:2501.01818) y RerouteGuard (arXiv:2601.21380).
Publicar un ataque exige divulgación responsable y acompañarlo de la defensa evaluada.


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from routedata import metrics`
- `from typing import Callable`
- `import numpy`
- `import random`
- `import re`
- `import string`

## ⚙️ Funciones globales (8)

### 📝 `typos(text, rate, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `Any` | — |
| `rate` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `case_whitespace_noise(text, seed)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `pad(text, seed, filler)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `Any` | — |
| `seed` | `Any` | — |
| `filler` | `Any` | — |

**Retorna:** `Any`

### 📝 `shuffle_options(options, answer_letter, seed)`

```python
    Baraja opciones de un MC y devuelve (nuevas_opciones, nueva_letra_correcta).
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `options` | `Any` | — |
| `answer_letter` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `Any`

### 📝 `paraphrase_requests(texts, backend_gen)`

```python
    Paráfrasis/traducción con un LLM local: backend_gen(prompt)->texto.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `texts` | `Any` | — |
| `backend_gen` | `Any` | — |

**Retorna:** `Any`

### 📝 `translate_requests(texts, lang, backend_gen)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `texts` | `Any` | — |
| `lang` | `Any` | — |
| `backend_gen` | `Any` | — |

**Retorna:** `Any`

### 📝 `gadget_search(router_score, queries, vocab, length, iters, seed)`

```python
    Búsqueda aleatoria/greedy de una secuencia de tokens que, antepuesta a cualquier
    consulta, MAXIMICE la puntuación de 'enviar al modelo caro' del router.
    
    router_score(lista_de_textos) -> array de P(strong) (black-box, solo salidas).
    Devuelve (gadget, score_medio_inicial, score_medio_final). Evalúa defensas con él;
    no lo uses contra servicios de terceros.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `router_score` | `Any` | — |
| `queries` | `Any` | — |
| `vocab` | `Any` | — |
| `length` | `Any` | — |
| `iters` | `Any` | — |
| `seed` | `Any` | — |

**Retorna:** `str`

### 📝 `ensure_attack_report(choice_clean, choice_attacked, C, Q)`

```python
    Resumen de impacto: tasa de cambio de decisión, inflación de coste y caída de calidad.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `choice_clean` | `Any` | — |
| `choice_attacked` | `Any` | — |
| `C` | `Any` | — |
| `Q` | `Any` | — |

**Retorna:** `Any`
