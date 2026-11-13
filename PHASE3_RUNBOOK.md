# Runbook · Fase 3 (DriftRoute, E4 · RQ4 · H4)

Todo el código de esta fase está probado **solo con datos sintéticos y un simulador** (12 + 3 pruebas, incluida una ejecución de la CLI de extremo a extremo con `MockBackend`). Nada se ha ejecutado con vLLM, GPU ni modelos reales: espera ajustes.

## 0. ¿Se puede hacer la Fase 3 antes de completar la Fase 2?

**Sí, con condiciones, y conviene empezar pronto**: DriftRoute es una serie temporal (6–9 meses) y el calendario es su cuello de botella, no el cómputo. El plan ya la preveía en paralelo. Pero cada snapshot debe ser **comparable** con el anterior, y eso crea dependencias con decisiones de la Fase 2:

| Decisión de Fase 2 | ¿Bloquea la Fase 3? | Qué hacer ahora |
|---|---|---|
| Plantillas de prompt (`generate/prompts.py`), niveles de esfuerzo (`efforts.py`) | **Sí, si cambian después de T0**: rompen la comparabilidad | Congélalas antes de `s00`. El manifiesto guarda `prompt_hash`; `check` avisa si difiere |
| Parámetros de generación (T=0,2, top_p=1, semilla) | **Sí** | Congélalos (en `configs/driftroute.yaml`) |
| Pool de modelos y revisiones HF | Parcial | El pool de `s00` puede ser pequeño (4–6 modelos). Los cambios posteriores son *eventos*, no ajustes |
| Juez LLM y su validación (puerta de la Fase 2) | **No, si usas solo tareas verificables** | `verifiable_only: true` (por defecto): MC, matemáticas y código no necesitan juez. Si luego añades tareas abiertas, **congela revisiones y prompt del juez** (campo `judge_freeze` del pool) |
| Escala de ConfigRoute (8 000 / 20 000), rejilla completa | No | DriftRoute usa una rejilla reducida y estable (efforts `nothink`, `think_mid`; presupuestos 500 y por defecto) |
| Partición privada del 10 % | Sí | Sepárala antes y pásala con `--exclude-ids` |
| Competencia por GPU | Práctica | Un snapshot de 3 000 consultas × ~15 configs son ≈ 45 000 instancias: ≈ 5 GPU-h con supuestos optimistas, ≈ 200 GPU-h con el ancla empírica de LLMRouterBench (pesimista). **Mide en el primer snapshot antes de comprometer ocho** |

**Riesgo principal:** si después cambias plantillas, rejilla, subconjunto o juez, tendrás que reiniciar la serie y perderás meses. Si no estás seguro de que la Fase 2 va a cambiar algo de eso, haz primero un **ensayo corto** (dos snapshots en semanas) y trátalo como piloto fuera de la serie.

## 1. Qué mide DriftRoute (y qué no)

Con pesos abiertos y revisión fijada, la «deriva» que puedes medir es: (a) nuevas revisiones o lanzamientos de modelos, (b) cambios de cuantización, motor de inferencia o hardware, (c) cambios de precio (GPU o proveedores de los mismos pesos), (d) altas y bajas de modelos en el pool, (e) deriva de **consultas** (lotes nuevos posteriores al corte). **No** mide cambios silenciosos de endpoints propietarios. Declara esa limitación en el artículo.

## 2. Requisitos previos
1. `python tests/test_synthetic.py && python tests/test_phase2.py && python tests/test_phase3.py`.
2. Pool con revisiones HF fijas (`configs/pool_openweights.yaml`; añade `judge_freeze` si habrá juez).
3. Tabla `queries` del esquema (de la Fase 1/2) y, si ya existe, el fichero JSONL con los `query_id` de la partición privada.
4. Decide el **embedding de consultas** definitivo para la evaluación. El kit usa TF-IDF+SVD solo como valor por defecto de pruebas (`routedata.drift.featurize_tfidf`); para publicar pasa una función propia a `DR.load_snaps(featurize=...)` con un modelo de embeddings **fijo** (cambiarlo entre ejecuciones confundiría la deriva con un cambio de representación).

## 3. Estructura de datos
    data/drift/subset/{queries.*, split.json, subset_manifest.json}   # se congela en `init`
    data/drift/s00/ s01/ ...                                          # configs, responses_*.jsonl, manifest_sXX.json
Los directorios de snapshot se llaman `s00`, `s01`, … (`s` + dígitos).

## 4. Paso a paso

### 4.1 Congelar el subconjunto (una sola vez)
    python -m generate.run_phase3 init --cfg configs/driftroute.yaml --drift-dir data/drift --queries data/queries/queries --exclude-ids privado.jsonl
Crea un subconjunto estratificado de `subset_queries` (3 000) y una **partición train/test fija por cluster** (50/50). No la regeneres nunca: los lotes nuevos irán solo a test.

### 4.2 Snapshots «placebo» (recomendado: s00 y s01 sin eventos)
Ejecuta dos snapshots con el **mismo pool** para medir el *suelo de ruido* (no determinismo de vLLM, batching, etc.). Todo cambio medido después debe superarlo.
    python -m generate.run_phase3 snapshot --drift-dir data/drift --snapshot s00 --pool pool_s00.yaml --date 2026-11-15
    python -m generate.run_phase3 snapshot --drift-dir data/drift --snapshot s01 --pool pool_s00.yaml --date 2026-11-16
    python -m generate.run_phase3 check --drift-dir data/drift
    python -m generate.run_phase3 report --drift-dir data/drift        # los «flips» entre s00 y s01 son el ruido base
Cada snapshot es reanudable (repite el comando si se corta), guarda el manifiesto (revisiones, versiones de vLLM/torch/GPU, precios derivados del throughput medido, hash de prompts y de consultas) y **reprecia** las configs con el tok/s medido.

### 4.3 Rutina mensual
1. ¿Hay evento? Regístralo con `new-pool` (genera `pool_sXX.yaml` y anota los eventos):
        python -m generate.run_phase3 new-pool --base-pool pool_s01.yaml --out pool_s02.yaml --snapshot s02 \
            --add "org/Modelo-14B=<commit>,14,1,1"          # modelo=revisión,size_b,thinking(0/1),tp
            --revision "Qwen/Qwen3-8B=<commit_nuevo>" \
            --price-factor "Qwen/Qwen3-8B=0.5" \
            --retire "google/gemma-2-9b-it"
            [--new-batch]                                      # evento de lote nuevo de consultas
2. `snapshot` con ese pool. Un cambio de **precio global de GPU** se hace editando `gpu.usd_per_hour` en el pool nuevo y anotándolo en `--note` (el CLI solo registra eventos por modelo).
3. **Lote nuevo de consultas:** pasa `--new-batch-file lote.jsonl --batch-id b1` en ese snapshot **y en todos los posteriores** (si no, el conjunto de consultas cambiaría y `check` avisará). El lote debe ser posterior al corte de datos de los modelos y seguir el esquema `queries`.
4. `check`, `report`. No modifiques prompts, rejilla, semilla ni juez durante la serie.

### 4.4 Evaluación E4
    python -m generate.run_phase3 eval --drift-dir data/drift
Genera `e4_results.csv` y `E4_REPORT.md`. Entrena en `s00` y evalúa en cada snapshot posterior sobre `test_ids`, con tres routers:
- **static:** entrenado en `s00`, sin actualizar (calidad y precios obsoletos; no puede elegir configs nuevas).
- **refit(f):** kNN reentrenado en el snapshot con una fracción `f` del tráfico de calibración (`f = 1` es la cota superior de reentrenar).
- **profile_online(f):** priores por perfil de configuración (precio, tamaño, thinking, esfuerzo, presupuesto) para configs nuevas + corrección online del sesgo de calidad y la escala de coste con feedback en sombra de una fracción `f`.

**Cómo leer el informe**
- `norm_gain` = (AIQ del router − AIQ de las configs individuales) / (techo − AIQ individuales). Cero = no mejora a elegir una config fija. **Mira siempre `ng_static` y `ng_full`:** una «recuperación» del 80 % es poco informativa si el reentrenado completo apenas supera a la mejor config fija.
- `rel_loss_static` = 1 − AIQ(static)/AIQ(refit f=1) → H4a (≥ 10 %).
- `f_needed_*` = fracción mínima de tráfico para recuperar el 80 % de la brecha estático→reentrenado; `NaN` = no se alcanza en la rejilla de `f` → H4b (≤ 1 %).
- Las filas con lote nuevo se pueden aislar con `evaluate_snapshot(..., subset="new")` para separar deriva de consultas de deriva del pool.

Ensayo en seco (sin GPU, sintético; sirve para ver el formato, **no es evidencia**):
    python -m generate.run_phase3 simulate

## 5. Puertas de paso
- `check` en ✅ en todos los snapshots (o los avisos están explicados por eventos).
- Suelo de ruido medido (placebo) y reportado; los eventos se interpretan solo por encima de él.
- ≥ 3 eventos reales de pool/precio/revisión (si no, complementar con *replays* sobre versiones antiguas de modelos abiertos).
- Cobertura: ≥ 6 snapshots antes de afirmar nada sobre tendencias.

## 6. Limitaciones conocidas (declararlas)
1. **Sin intervalos de confianza por evento** en `eval`: añade *bootstrap* sobre consultas de test antes de publicar (pendiente en el kit).
2. Pocos snapshots → baja potencia estadística para H4; trata H4 como estudio de casos por evento, no como prueba general.
3. Los routers (kNN, perfil+online) son **referencias simples**, no CCR completo. La recuperación del 80 % se compara con un reentrenado kNN que puede ser débil.
4. Feedback «en sombra» sobre todas las configs: es optimista frente a un despliegue donde solo observas la config elegida (bandido). La variante con feedback parcial no está implementada.
5. Los eventos de precio por modelo son factores sobre el precio derivado; el precio real de la GPU varía por contrato y uso (`utilization`).
6. El simulador usa una dinámica inventada; sus cifras no sustentan ninguna conclusión.

## 7. Lo que NO verifiqué
- Ejecución con vLLM real, `nvidia-smi` y versiones de paquetes en tu entorno (`capture_environment` está probado solo sin GPU).
- Que los snapshots reales sean estables entre ejecuciones (de ahí el placebo).
- Los esquemas de datasets externos (ver runbook de la Fase 2).
