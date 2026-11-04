# Pipeline Resumen - routedata

## Estado: ✅ Funcional (3 datasets)

El pipeline carga y combina 3 de los 4 datasets disponibles:

| Dataset | Queries | Configs | Responses | Estado |
|---------|---------|---------|-----------|--------|
| RouterBench | 73,008 | 20 | 803,088 | ✅ |
| SPROUT | 44,241 | 13 | 575,133 | ✅ (esquema corregido) |
| LLMRouterBench | 27,203 | 40 | 1,088,120 | ✅ |
| R2-Bench | - | - | - | ⚠️ Timeout (demasiado grande) |
| **TOTAL** | **144,452** | **75** | **2,466,341** | |

## Corrección Aplicada

### Problema
El adaptador de SPROUT generaba un esquema diferente:
- `id` en lugar de `query_id`
- `prompt` en lugar de `text`
- `gold` en lugar de `answer`

Esto rompía la integridad referencial con las tablas `responses`.

### Solución
En `routedata/adapters/sprout.py`, se normalizó el esquema:
```python
queries = queries.rename(columns={"id": "query_id", "prompt": "text", "gold": "answer"})
config_df = config_df.rename(columns={"id": "config_id", "name": "model", "kind": "kind"})
```

## Uso

```bash
# Ejecutar pipeline completo (solo 3 datasets por timeout)
source .venv/bin/activate
python -m routedata.pipeline load

# O usar el script de regeneración
python regenerate_combined.py
```

## Archivos Generados

```
data/processed/unified/
├── queries_combined.parquet    # 88 MB - 144,452 filas
├── configs_combined.parquet    # 10 KB - 75 filas
├── responses_combined.parquet  # 456 MB - 2,466,341 filas
├── manifest.json               # Metadatos
└── (parquets individuales por dataset)
```

## Esquema Unificado

### `queries`
- `query_id`: ID único de la query
- `text`: Texto de la pregunta
- `answer`: Respuesta correcta (gold)
- `domain`: Dominio (opcional)
- `category`: Categoría (opcional)
- `language`: Idioma (opcional)
- `source_dataset`: Nombre del dataset origen

### `configs`
- `config_id`: ID único de la configuración
- `model`: Nombre del modelo
- `kind`: "candidate" o "router"
- `price_in`, `price_out`: Costo por token
- `price_source`, `price_date`: Fuente y fecha del precio

### `responses`
- `id`: ID único de la respuesta
- `query_id`: FK a queries
- `config_id`: FK a configs
- `correct`: booleano (si la respuesta fue correcta)
- `score`: score normalizado [0, 1]
- `response`: Texto de la respuesta
- `latency_ms`: Latencia (si disponible)

## Agente de Routing

El paquete `routedata.agents` incluye:

### Base Classes (`base.py`)
- `RoutingAgent`: Clase base para todos los agentes
- `RoutingDecision`: Resultado de la decisión de routing
- `Candidate`: Modelo candidato con metadatos
- `RoutingStrategy`: Enum de estrategias disponibles

### Baselines (`baseline.py`)
- `RandomAgent`: Selección aleatoria (baseline)
- `CostAgent`: Siempre el más barato
- `QualityAgent`: Siempre el de mejor calidad
- `LatencyAgent`: Siempre el más rápido
- `HybridAgent`: Combina costo + calidad + latencia
- `HeuristicAgent`: Usa heurísticas (lenguaje, dominio, complejidad)

### Uso
```python
from routedata.agents import create_agent, Candidate

# Crear agente
agent = create_agent("heuristic")

# Definir candidatos
candidates = [
    Candidate(name="gpt-4o", cost_per_1k_tokens=3.0, quality_score=0.9),
    Candidate(name="mistral-7b", cost_per_1k_tokens=0.3, quality_score=0.7),
]

# Tomar decisión
decision = agent.decide("What is the capital of France?", candidates)
print(decision.chosen_model)  # gpt-4o
print(decision.reasoning)     # "Heuristic: language=english, domain=general..."
```

## Próximos Pasos

1. **R2-Bench**: Procesar por separado o optimizar (30 CSVs, timeout)
2. **Eval pipeline**: Comparar agentes usando el dataset unificado
3. **Feature extraction**: Agregar features a `queries` (embeddings, dominio)
4. **Model training**: Entrenar router aprendido con XGBoost
