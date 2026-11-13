# Estado del Proyecto: Routedata

**Fecha de verificación:** 2026-11-09
**Estado:** ✅ Ingestión validada y evaluación local de Fase 1 ejecutada; reproducción de routers publicados y RouterArena pendientes.

## Resumen

- Esquema de ingesta normalizado para RouterBench, SPROUT y LLMRouterBench.
- Dataset unificado regenerado; sus recuentos se contrastan con `manifest.json`.
- Paquete `routedata.agents` contiene seis agentes baseline.
- La evaluación local aporta evidencia preliminar para Fase 1, pero no completa sus entregables externos ni demuestra las puertas de las fases posteriores de `PLAN.md`.

## Fase 1 · Auditoría y evaluación multiset

- ✅ Auditoría de los 20 modelos oficiales y 15 datasets de rendimiento; 11.480 filas completas consulta-dataset en matriz compartida de 20 candidatos.
- ✅ Ejecutados seis routers/baselines locales en cinco splits prompt-level 70/30 con las semillas del PLAN; cada router se ajusta solo con train. No son implementaciones reproducidas de cinco routers publicados.
- Calidad macro por dataset, media ± desviación entre semillas: random 0,478 ± 0,013; best-single 0,718 ± 0,007; length-bucket 0,725 ± 0,011; TF-IDF kNN 0,697 ± 0,014; TF-IDF classifier 0,592 ± 0,015; regresor TF-IDF 0,707 ± 0,011; Dataset Oracle 0,738 ± 0,014; oráculo por consulta 0,916 ± 0,010.
- La brecha entre el mejor baseline y Dataset Oracle es pequeña (~0,013); el gap al oráculo por consulta permanece (~0,214). Recall-failure pooled entre consultas resolubles: ~0,216. Compatible cualitativamente con el hallazgo del PLAN, pero no es reproducción bibliográfica completa.
- Auditoría detectó un score agregado para `mbpp/test/665`, excluido del análisis completo, y 1.001 consultas MMLU-Pro test_1000 frente a las 1.000 de la tabla oficial; la diferencia se conserva y documenta.
- Artefactos: `outputs/phase1/protocol_audit.json` y `outputs/phase1/llmrouterbench_15datasets.json`. El experimento anterior de ArenaHard queda en `outputs/phase1/llmrouterbench_arenahard.json` como piloto aparte.
- Split contrastado con el loader upstream: por dataset y prompt, sin fuga entre train/test. Este trabajo usa la proporción 70/30 y cinco semillas del PLAN; el API upstream es configurable y sus defaults públicos son distintos en algunos adaptadores.
- RouterArena `sub_10` (809 preguntas; commit `a4a062ce3313b56bb09c042e1bc37b61d34e3bd8`) está descargado como holdout, preservando `Context` y `Options`, marcado `eval_only`. No hay respuestas de modelos para puntuarlo; la evaluación externa de calidad sigue pendiente de inferencia en GPU. No se ha usado RouterArena para entrenar ni ajustar.

## Fase 0 · Infraestructura

- ✅ SHA de commit fijados para los 9 modelos del pool y los 3 jueces; verificados contra la API de Hugging Face el 2026-11-13 (`revision_checked_at`).
- ✅ `python tests/test_synthetic.py`: 7 pruebas pasan; última suite completa (2026-11-13): 53 pasan.
- ✅ Añadido `scripts/benchmark_vllm.py` y el target `make phase0-measure` para medir throughput agregado, derivar precios y guardar resultados en JSONL.
- ⏸️ Gate aún abierto: el host disponible es un Mac M2 Max, no tiene NVIDIA/CUDA y no tiene vLLM; no se han medido tok/s reales ni derivado precios. El coste GPU A100 de `pool_openweights.yaml` continúa siendo un supuesto.

Para completar el gate, ejecutar la medición en Linux/NVIDIA con una tarifa verificada del proveedor; repetirla para cada modelo que vaya a usarse en Fase 1 y conservar `outputs/phase0/throughput.jsonl` como evidencia.

## Dataset unificado

**Ubicación:** `data/processed/unified/`

| Fuente | Queries | Configs | Responses |
|---|---:|---:|---:|
| RouterBench | 73.008 | 11 | 803.088 |
| SPROUT | 44.241 | 13 | 575.133 |
| LLMRouterBench | 27.203 | 40 | 548.059 |
| **Total** | **144.452** | **64** | **1.926.280** |

El manifiesto y las filas Parquet coinciden. Hay cero IDs de query/config duplicados, cero claves foráneas huérfanas y cero claves de respuesta duplicadas.

### Procedencia y calidad de datos

- LLMRouterBench aporta pares consulta-modelo observados, no el producto cartesiano completo. El recuento previo de 1.088.120 incluía 540.061 filas sin observación; se retiraron.
- 4.327 respuestas de LLMRouterBench tienen `label_source="aggregate_fallback"`: su score es agregado al archivo, no una etiqueta independiente por ejemplo.
- 154 respuestas de RouterBench no traen score. Se conservan con `correct=NaN`, no se convierten en errores.
- 13 contadores negativos de salida de LLMRouterBench se normalizaron a ausentes. Para esos registros el coste publicado es 0 USD.
- Ninguna respuesta histórica tiene `latency_s`; no usar este conjunto para comparar latencias.
- R2-Bench tiene ficheros crudos presentes, pero no está incluido en el unificado. `make validate` revisa una muestra de sus CSV, no integra ni valida todas sus filas.

### Esquema almacenado

- **Queries:** `query_id`, `text`, `answer`, `domain`, `category`, `source_dataset`, además de campos opcionales.
- **Configs:** `config_id`, `model`, `revision`, `effort`, `budget`, precios y fuente/fecha de precio.
- **Responses:** `query_id`, `config_id`, `run_id`, `snapshot_id`, tokens, `latency_s`, `correct`, `cost_usd` opcional y `label_source`.

## Validación reproducible

```bash
make phase1             # regenera RouterBench, SPROUT y LLMRouterBench
make validate           # valida estructuralmente los datasets crudos
make validate-unified   # valida manifiesto, recuentos, IDs, FK y métricas
make test               # suite completa
```

## Paquete routedata

Incluye esquema y validación, pipeline de carga, estadísticas, métricas, splits, precios, adaptadores y agentes baseline.

### Ejemplo de agente

```python
from routedata.agents import Candidate, create_agent

agent = create_agent("heuristic")
candidates = [
    Candidate(name="qwen2.5-7b-instruct", cost_per_1k_tokens=0.05, quality_score=0.62, latency_ms=180),
    Candidate(name="qwen3-30b-a3b", cost_per_1k_tokens=0.30, quality_score=0.88, latency_ms=900),
]
decision = agent.decide("Explain how transformers work.", candidates)
print(decision.chosen_model, decision.reasoning)
```

Agentes disponibles: `random`, `cost`, `quality`, `latency`, `hybrid` y `heuristic`.

## Pendientes

### Próximos
- [ ] Integrar R2-Bench en un flujo separado o por lotes, preservando IDs, scores ausentes y su procedencia.
- [ ] Reproducir al menos cinco métodos publicados del benchmark con sus implementaciones/configuraciones y comparar con las baselines locales.
- [ ] Generar resultados para los candidatos y evaluar calidad en RouterArena `sub_10` sin entrenar ni ajustar con su contenido.
- [ ] Validar agentes con datos reales y documentar métricas.
- [ ] Completar flujos de generación/evaluación de Fases 2–5 y comprobar sus criterios de paso.

### Más adelante
- Entrenar y comparar routers aprendidos, añadir features (embeddings y complejidad), ejecutar A/B testing y estudiar personalización/RL.

## Conclusión

La ingesta y la integridad estructural de los tres datasets están verificadas localmente. La aptitud para experimentos concretos depende de filtrar las etiquetas agregadas o ausentes y de completar las reproducciones y gates del plan; el dataset todavía no constituye evidencia de que dichas fases se hayan superado.
