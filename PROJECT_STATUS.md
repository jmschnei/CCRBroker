# Estado del Proyecto: Routedata

**Fecha:** 2026-11-03
**Estado:** ✅ FUNCIONAL

---

## Resumen Ejecutivo

El paquete `routedata` está completo y funcional para el entrenamiento de modelos de routing. Hemos:

1. ✅ Normalizado el esquema de datos de SPROUT para unificarlo con RouterBench y LLMRouterBench
2. ✅ Generado un dataset unificado con 144,452 queries, 75 configs y 2.4M+ responses
3. ✅ Implementado el módulo `routedata.agents` con 6 agentes de baseline
4. ✅ Validado la integridad referencial completa

---

## Dataset Unificado

**Ubicación:** `data/processed/unified/`

### Estadísticas
- **Queries:** 144,452 únicas
- **Configurations:** 75 (modelos × parámetros)
- **Responses:** 2,466,341 evaluados
- **Sources:** SPROUT, RouterBench, LLMRouterBench

### Esquema Normalizado

**queries_combined.parquet**
```python
{
    "query_id": str,      # ID único
    "text": str,          # Texto de la query
    "answer": str,        # Respuesta de referencia (o "")
    "category": str,      # Categoría
    "difficulty": str,    # Dificultad
    "source": str         # SPROUT | RouterBench | LLMRouterBench
}
```

**configs_combined.parquet**
```python
{
    "config_id": str,     # ID único
    "model": str,         # Nombre del modelo
    "temperature": float,
    "max_tokens": int,
    "source": str
}
```

**responses_combined.parquet**
```python
{
    "query_id": str,      # FK → queries
    "config_id": str,     # FK → configs
    "score": float,       # 0.0 - 1.0
    "reward": float,      # Reward (si aplica)
    "latency_ms": float   # Latencia
}
```

### Integridad Referencial
```bash
✓ 0 query_ids faltantes
✓ 0 config_ids faltantes
✓ REFERENTIAL INTEGRITY CHECK PASSED
```

---

## Paquete routedata

### Estructura
```
routedata/
├── __init__.py
├── schema.py           # Esquema normalizado
├── validate.py         # Validación de integridad
├── pipeline.py         # Pipeline de carga
├── stats.py            # Estadísticas
├── metrics.py          # Métricas
├── splits.py           # Splits de datos
├── pricing.py          # Precios
├── baselines.py        # Baselines básicos
├── conformal.py        # Inferencia conformal
├── download.py         # Descarga de datasets
│
├── adapters/           # Adaptadores por dataset
│   ├── base.py
│   ├── sprout.py       # ✅ Normalizado
│   ├── routerbench.py  # ✅
│   ├── llmrouterbench.py # ✅
│   ├── r2bench.py      # ⚠️ Pendiente optimización
│   └── routerarena.py
│
└── agents/             # 🆕 NUEVO: Agentes de routing
    ├── __init__.py
    ├── base.py         # Interface base
    └── baseline.py     # 6 agentes de baseline
```

### Uso Básico

```python
from routedata.agents import create_agent, Candidate

# Crear agente
agent = create_agent("heuristic")

# Crear candidatos
candidates = [
    Candidate(model="gpt-4o", quality=0.9, cost=5.0, latency_ms=800),
    Candidate(model="claude-3-opus", quality=0.95, cost=15.0, latency_ms=1000),
    Candidate(model="mistral-7b", quality=0.7, cost=0.5, latency_ms=200),
]

# Rutear
decision = agent.route("Query text here", candidates)
print(f"Selected: {decision.selected_model}")
print(f"Score: {decision.score:.3f}")
print(f"Reasoning: {decision.reasoning}")
```

### Agentes Disponibles

| Agente | Descripción | Caso de uso |
|--------|-------------|-------------|
| `random` | Selección aleatoria | Baseline |
| `cost` | Más barato | Optimización de costos |
| `quality` | Mejor calidad | Máxima calidad |
| `latency` | Más rápido | Tiempo real |
| `hybrid` | Costo + calidad | Balance |
| `heuristic` | Heurística de complejidad | **Recomendado** |

---

## Problemas Resueltos

### 1. Inconsistencia de Esquema (RESUELTO ✅)
**Problema:** SPROUT usaba `id, prompt, gold` mientras los otros usaban `query_id, text, answer`

**Solución:** Normalizamos el adaptador de SPROUT en `routedata/adapters/sprout.py`

**Resultado:** Esquema unificado para todos los datasets

### 2. Integridad Referencial (RESUELTO ✅)
**Problema:** 11,951 query_ids y 3 config_ids faltantes en responses

**Solución:** Regeneramos los archivos combinados con el esquema corregido

**Resultado:** 0 IDs faltantes, integridad completa

### 3. R2-Bench Timeout (PENDIENTE ⚠️)
**Problema:** El pipeline falla al procesar R2-Bench (>10s timeout)

**Estado:** Excluido del pipeline automático por ahora

**Solución propuesta:**
- Optimizar el pipeline para R2-Bench
- O procesar R2-Bench en un paso separado

---

## Próximos Pasos

### Inmediatos
- [ ] **Integrar R2-Bench** en el pipeline (optimizar o proceso separado)
- [ ] **Validar heurísticas** en HeuristicAgent con datos reales
- [ ] **Testear agentes** con datos del dataset unificado

### Mediano plazo
- [ ] **Entrenar modelo de routing aprendido**
  - Ej: XGBoost, LightGBM
  - Features: complejidad, longitud, keywords
  - Target: config_id óptimo
- [ ] **Agregar features adicionales**
  - Embeddings de queries
  - Métricas de complejidad
  - Metadata de contexto

### Largo plazo
- [ ] **A/B testing** de agentes vs modelo aprendido
- [ ] **Reinforcement learning** para routing
- [ ] **Personalización** por dominio/user

---

## Comandos Útiles

### Verificar datos
```bash
cd /Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker

# Ver estadísticas
.venv/bin/python -c "
import pandas as pd
queries = pd.read_parquet('data/processed/unified/queries_combined.parquet')
print(f'Queries: {len(queries):,}')
print(queries['category'].value_counts())
"

# Ver integridad
.venv/bin/python -m routedata.validate
"

### Regenerar dataset combinado
```bash
.venv/bin/python regenerate_combined.py
```

### Usar agentes
```python
from routedata.agents import create_agent, Candidate

agent = create_agent("heuristic")
candidates = [
    Candidate(model="gpt-4o", quality=0.9, cost=5.0, latency_ms=800),
    Candidate(model="mistral-7b", quality=0.7, cost=0.5, latency_ms=200),
]
decision = agent.route("Your query here", candidates)
```

---

## Conclusiones

El proyecto está **listo para usar** en entrenamiento de modelos de routing:

✅ Dataset unificado con esquema consistente
✅ Integridad referencial garantizada
✅ API limpia y documentada
✅ Baselines implementados
✅ Estructura modular y extensible

**Listo para:**
1. Entrenar modelos de routing aprendidos
2. Evaluar baselines vs modelo aprendido
3. Experimentar con diferentes estrategias de routing
4. Personalizar por dominio o caso de uso
