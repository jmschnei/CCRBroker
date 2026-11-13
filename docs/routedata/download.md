# 📄 download.py

**Ruta relativa:** `routedata/download.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/routedata/download.py`

## 📖 Descripción del módulo

Descarga de datasets reales desde HuggingFace.

Uso:
    python -m routedata.download --dataset all
    python -m routedata.download --dataset routerbench
    python -m routedata.download --dataset sprout
    python -m routedata.download --dataset r2bench
    python -m routedata.download --dataset llmrouterbench
    python -m routedata.download --dataset all --force  # re-descarga aunque ya existan


## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from pathlib import Path`
- `import argparse`
- `import hashlib`
- `import json`
- `import logging`
- `import os`
- `import subprocess`
- `import sys`
- `import urllib.request`

## ⚙️ Funciones globales (5)

### 📝 `_sha256(path, chunk_size)`

```python
    Calcula SHA-256 de un archivo.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `path` | `Any` | — |
| `chunk_size` | `Any` | — |

**Retorna:** `Any`

### 📝 `download_file(url, dest)`

```python
    Descarga un archivo desde una URL. Devuelve True si se descargó correctamente.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `url` | `Any` | — |
| `dest` | `Any` | — |

**Retorna:** `Any`

### 📝 `download_dataset(name, force)`

```python
    Descarga un dataset completo. Devuelve lista de archivos descargados.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `name` | `Any` | — |
| `force` | `Any` | — |

**Retorna:** `Any`

### 📝 `download_all(force)`

```python
    Descarga todos los datasets.
```

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `force` | `Any` | — |

**Retorna:** `Any`

### 📝 `main()`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|

**Retorna:** `Any`
