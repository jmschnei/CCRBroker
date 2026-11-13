# Runbook · Fase 2 (ConfigRoute)

Todo el código se ha probado solo con datos sintéticos y un simulador. Los pasos que tocan Hugging Face, vLLM y GPU son **nuevos y sin ejecutar en datos reales**: espera tener que ajustar columnas, parámetros de vLLM o plantillas de chat.

## Paso 0 · Comprobaciones previas (≈ 1 h)
1. `python tests/test_synthetic.py && python tests/test_phase2.py` (deben pasar 7 + 5 pruebas).
2. En `configs/pool_openweights.yaml`: sustituye cada `PIN_ME` por el commit HF exacto; ajusta `tp` (GPUs por modelo), `gpu.usd_per_hour` y elige 4 modelos para el piloto (1 pequeño, 2 medianos —uno con thinking—, 1 grande si cabe).
3. `python -m generate.build_queries inspect mmlu_pro` (y `math500`, `gpqa_diamond`, `humaneval`): compara columnas con `SOURCES` en `generate/build_queries.py` y corrige. GPQA requiere aceptar condiciones en HF; verifica licencias.
4. Comprueba que el chat template de tus modelos admite `enable_thinking` (familia Qwen3) o define cómo activar/desactivar el razonamiento en los demás. Los modelos sin modo thinking solo reciben `nothink`.

## Paso 1 · Consultas del piloto
    python -m generate.run_phase2 build-queries --cfg configs/configroute_pilot.yaml --out data/queries --open-ended abiertas.jsonl
`abiertas.jsonl` (OpenHermes/ArenaHard u otras con licencia verificada) debe seguir el esquema `queries` (`routedata/schema.py`; `meta.task_type = "open"`, `answer` = referencia).

## Paso 2 · Piloto (≈ 1 000 consultas)
    python -m generate.run_phase2 pilot --pool configs/pool_openweights.yaml --cfg configs/configroute_pilot.yaml --queries data/queries --out data/pilot
Es reanudable: si se corta, repite el mismo comando. Qué vigilar en el log: truncados masivos (sube `max_model_len` o presupuesto), respuestas sin «Final answer:» (el verificador las puntúa 0: revisa la plantilla de prompt antes de culpar al modelo).

## Paso 3 · Etiquetas del juez y auditoría humana
    python -m generate.run_phase2 judge --run-dir data/pilot --pool configs/pool_openweights.yaml
    python -m generate.run_phase2 audit-export --run-dir data/pilot --n 150
Reparte `data/pilot/audit/audit_sheet.csv` (ciega: sin modelo ni nota del juez) a **dos anotadores**; rellenan `human_score_a/b` (0–1) y:
    python -m generate.run_phase2 audit-import --run-dir data/pilot --sheet <hoja_rellenada.csv>
Guarda `audit_key_PRIVATE.csv` fuera del alcance de los anotadores. Las etiquetas humanas prevalecen sobre las del juez.
Jueces: pon en `pool.judges` modelos de familias distintas y, a ser posible, fuera del pool evaluado; en el `judge` real fija revisiones concretas (el CLI usa `main` por defecto: cámbialo).

## Paso 4 · Informe y decisión
    python -m generate.run_phase2 report --run-dir data/pilot --pool configs/pool_openweights.yaml --cfg configs/configroute_pilot.yaml --n-full 8000
Genera `REPORT.md`: acierto, tokens, truncados, cumplimiento de longitud, tok/s agregado y **GPU-horas y USD proyectados** por configuración, más las puertas de paso.

### Puertas de la Fase 2 (de `PLAN.md`)
- Juez vs humano: Pearson ≥ 0,8 y κ ≥ 0,7 (κ entre anotadores también; si es bajo, el problema son las instrucciones de anotación, no el juez).
- Cumplimiento de longitud ≥ 80 % en los modelos grandes (si no, el presupuesto no es una palanca fiable).
- Coste proyectado dentro de tu presupuesto. **Si no cabe, reduce primero configuraciones y modelos, no el rigor estadístico.**

### Decisiones que debes tomar tras el piloto
1. **Escala:** punto de partida recomendado, 8 000 consultas (no 20 000); amplía solo si el informe lo permite.
2. **Rejilla:** descarta configuraciones dominadas (mismo acierto, más tokens) antes de la escala completa.
3. **Dificultad empírica:** con las tasas de acierto del piloto usa `assign_empirical_difficulty` para estratificar la muestra completa de forma homogénea entre fuentes.
4. **Multilingüe:** traduce con `generate/multilingual.py` (traductor local fuera del pool), revisa humanamente las marcadas + muestra aleatoria, y añade consultas **nativas** para no medir solo «inglés traducido».
5. **Partición privada del 10 %** para el leaderboard: sepárala antes de generar nada y no la uses para ajustar.

## Paso 5 · Escala completa (tras superar las puertas)
Mismo flujo con `--cfg` de escala (copia `configroute_pilot.yaml`, sube `n_pilot`, añade `think_high` y presupuestos) y `audit_n` = 500. Mantén `run_id` y revisiones fijos; cualquier cambio de revisión pertenece a DriftRoute, no a ConfigRoute.

## Limitaciones conocidas
- «Esfuerzo» = *budget forcing* en dos pasadas (definición operativa); distinto del esfuerzo nativo de modelos propietarios. Decláralo en el artículo.
- El simulador (`MockBackend`) no representa comportamiento real: las cifras del informe de prueba no significan nada.
- El coste en el piloto sale con precios 0 en `configs`; `report` proyecta GPU-horas y USD con tus parámetros de GPU. Para usar `to_matrices` con coste real, deriva precios por token con `routedata/pricing.py` a partir del tok/s medido y actualiza `configs`.
