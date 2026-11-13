# Pipeline Resumen - routedata

## Estado: ✅ Regenerado y validado (3 datasets; 2026-11-09)

El pipeline carga y combina 3 de los 4 datasets disponibles:

| Dataset | Queries | Configs | Responses | Estado |
|---------|---------|---------|-----------|--------|
| RouterBench | 73,008 | 11 | 803,088 | ✅ (configs únicas entre 0-shot y 5-shot) |
| SPROUT | 44,241 | 13 | 575,133 | ✅ (esquema corregido) |
| LLMRouterBench | 27,203 | 40 | 548,059 | ✅ (solo pares consulta-modelo observados) |
| R2-Bench | — | — | — | ⚠️ Crudo presente; excluido del unificado actual |
| **TOTAL** | **144,452** | **64** | **1,926,280** | |

### Calidad y procedencia de labels

- 4,327 labels de LLMRouterBench usan `aggregate_fallback`; no son labels por instancia.
- 154 respuestas de RouterBench no tienen score y se conservan como `correct=NaN`, no como cero.
- 13 contadores negativos de tokens en LLMRouterBench se normalizaron a ausentes; sus costes reportados son 0 USD.
- `latency_s` está ausente en todas las respuestas históricas; estos datos no permiten comparar latencia.
- `make validate-unified` coteja el manifiesto, recuentos, unicidad de IDs, claves foráneas, fuentes y métricas. Resultado actual: 0 duplicados y 0 referencias huérfanas.

### Reconciliación del recuento anterior

El número antiguo de LLMRouterBench (1,088,120) incluía 540,061 filas sin observación,
generadas al completar el producto cartesiano consulta × modelo. El adaptador actual
conserva las 548,059 respuestas observadas. RouterBench comparte los mismos 11 modelos
entre sus dos splits; se guarda una sola fila por configuración, no 22.

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
# Regenerar las tablas unificadas de los 3 datasets incluidos
source .venv/bin/activate
python regenerate_combined.py

# Validar datasets crudos y Parquet unificados
python -m routedata.validate
python -m routedata.validate --unified
```

## Archivos Generados

```
data/processed/unified/
├── queries_combined.parquet    # 144,452 filas
├── configs_combined.parquet    # 64 filas
├── responses_combined.parquet  # 1,926,280 filas
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
- `revision`, `effort`, `budget`: metadatos de configuración (pueden estar vacíos/NaN en datasets heredados)
- `price_in`, `price_out`: precio por millón de tokens cuando existe; cero en adaptadores de coste final
- `price_source`, `price_date`: Fuente y fecha del precio
- `hardware`, `source_dataset`: hardware o dataset de origen

### `responses`
- `query_id`: FK a queries
- `config_id`: FK a configs
- `run_id`, `snapshot_id`: identificadores de ejecución/snapshot
- `response_text`: respuesta textual, vacía para fuentes que no la contienen
- `in_tokens`, `out_tokens`, `reasoning_tokens`: contadores de tokens; pueden faltar
- `correct`: calidad normalizada [0, 1], nullable
- `label_source`: procedencia de la etiqueta, incluyendo `aggregate_fallback`
- `cost_usd`: coste final por respuesta cuando la fuente lo proporciona
- `latency_s`: latencia en segundos; actualmente ausente en todos estos datos históricos

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

1. **R2-Bench**: decidir e implementar su integración por separado; está excluido de la regeneración actual por tamaño
2. **Eval pipeline**: Comparar agentes usando el dataset unificado
3. **Feature extraction**: Agregar features a `queries` (embeddings, dominio)
4. **Model training**: Entrenar router aprendido con XGBoost
