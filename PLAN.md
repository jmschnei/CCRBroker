# Plan de benchmarks y datasets para evaluar routing (pesos abiertos, por fases)

Fecha de verificación de fuentes: 21-oct-2026. Recursos: solo modelos de pesos abiertos en GPU propia o alquilada.

## 1. Decisión central: reutilizar primero, crear solo lo que falta

| Necesidad (RQ/H) | ¿Existe? | Decisión |
|---|---|---|
| Línea base y brecha con el oráculo, modelo único (RQ1, RQ2) | Sí: **LLMRouterBench** (pool ligero de 20 modelos de ~7–9B abiertos, 11 480 prompts, 229 600 instancias), RouterBench, SPROUT | **Reutilizar.** El pool ligero es justo del tipo que puedes autoalojar |
| Evaluación externa comparable (RQ1, RQ6 parcial) | Sí: **RouterArena** (8 400 preguntas; splits `sub_10` 809, `full` 8 400, `robustness` 420) | **Reutilizar solo para evaluar.** Prohibido entrenar con él |
| Configuraciones modelo × esfuerzo × longitud (RQ3) | Parcial: **R2-Bench** (30 968 consultas, 15 LLM, 16 presupuestos de tokens). No verifiqué que los datos sean públicos | **Crear ConfigRoute** (replicando su protocolo) o reutilizar R2-Bench si es público |
| Esfuerzo de razonamiento (RQ3) | No | **Crear** (budget forcing en dos pasadas, ver `generate/efforts.py`) |
| Deriva temporal (RQ4) | No | **Crear DriftRoute** (con pesos abiertos: deriva de pool, no de API cerrada) |
| Agentes con coste de cambio (RQ5) | No | **Crear AgentRoute** |
| Perturbaciones y ataques (RQ6) | Parcial: split `robustness` de RouterArena (420) | **Crear AdvRoute**; usar el split de RouterArena como contraste |
| Multilingüe (RQ8) | No en los benchmarks de routing revisados | **Crear** subconjuntos ES/DE/FR dentro de ConfigRoute |
| Garantías de riesgo (RQ7) | Se calcula sobre cualquiera de los anteriores | Sin dataset nuevo: `routedata/conformal.py` |

Catálogo detallado (con qué verifiqué y qué no): `catalog/dataset_catalog.yaml`.

## 2. Fases y criterios de paso (go/no-go)

**Fase 0 · Infraestructura (semana 1).** Entorno con vLLM, fijar revisiones HF de cada modelo (`PIN_ME` → commit), medir tok/s por modelo con carga real, derivar precios (`routedata/pricing.py`), repetir `python tests/test_synthetic.py` en tu máquina.
*Paso a Fase 1 si:* las pruebas pasan y tienes tok/s medidos.

**Fase 1 · Reutilización, solo selección de modelo (semanas 1–3).** Ingerir LLMRouterBench (pool ligero), RouterBench y SPROUT al esquema unificado; reproducir ≥5 routers con las 5 semillas y el split 70/30 de LLMRouterBench; añadir baselines triviales y descomposición de la brecha. Evaluar en RouterArena (`sub_10` primero).
*Entregables:* E1, E2. *Paso si:* reproduces cualitativamente los hallazgos de LLMRouterBench (muchos routers parecidos, brecha al oráculo por *recall*); si no, investigar antes de crear datos.

**Fase 2 · ConfigRoute (semanas 3–8).** Piloto de 1 000 consultas para calibrar GPU-horas, luego la escala completa (ver §4). Etiquetado: verificadores primero, jueces en conjunto para tareas abiertas, 500 ítems humanos.
*Entregables:* E3, E7, E8. *Paso si:* el juez alcanza Pearson ≥ 0,8 y κ ≥ 0,7 frente a humanos, y el cumplimiento de la restricción de longitud es ≥ 80 % en los modelos grandes.

**Fase 3 · DriftRoute (semanas 6–16, en paralelo).** Snapshots mensuales sobre 3 000 consultas, con manifiesto. *Entregable:* E4. *Paso si:* ocurren al menos 3 eventos reales de pool/precio/revisión (si no, complementar con *replays* de versiones antiguas de modelos abiertos).

**Fase 4 · AgentRoute (semanas 8–14).** Adaptadores de entorno (τ²-Bench, ScienceWorld, subconjunto de SWE-bench, QA multi-salto), políticas mixtas, registro de prefill por cambio de modelo. *Entregable:* E5. *Paso si:* un modelo local puede completar ≥ 20 % de las tareas (si no, el contrafactual no discrimina).

**Fase 5 · AdvRoute (semanas 12–16).** Perturbaciones benignas, gadget black-box, detector. *Entregable:* E6.

## 3. Qué hay en el kit (`routedata/` y `generate/`)

- `routedata/schema.py`: esquema unificado (queries / configs / responses), validación, conversión a matrices Q y C.
- `routedata/adapters/`: RouterBench (pickle, con aviso de seguridad), RouterArena (con bloqueo `eval_only`), adaptador genérico ancho→largo.
- `routedata/metrics.py`: Gain@b, Gap@O, PerfGain/CostSave (LLMRouterBench), Pareto/hipervolumen/ParetoDist, AUDC/QNC (deferral curve, formulación de R2-Router), AIQ (aproximada), consistencia, inflación de coste, ECE/Brier, riesgo-cobertura.
- `routedata/stats.py`: bootstrap, McNemar exacto, Holm, Benjamini–Hochberg, κ de Cohen.
- `routedata/splits.py`: iid, por clusters, leave-domain-out, leave-model-out, temporal, prequential, semillas de LLMRouterBench.
- `routedata/conformal.py`: control de riesgo conformal (CRC) con umbral por estrato (núcleo de CCR/E7).
- `routedata/pricing.py`: precio por token derivado de GPU-hora y tok/s medidos, con escenarios ±50 %.
- `generate/`: backends (vLLM y simulador), prompts con restricción de longitud, esfuerzo por *budget forcing*, verificadores, juez en conjunto y muestreo humano, `configroute.py` (reanudable), `driftroute.py`, `agentroute.py`, `advroute.py`.
- `configs/`, `catalog/`, `estimate_budget.py`, `tests/test_synthetic.py`.

## 4. Presupuesto de cómputo: la incertidumbre es grande

`estimate_budget.py` con supuestos optimistas (2 500 tok/s de decodificación agregados, 600 tokens de salida medios, 1 GPU) da ≈ 144 GPU-horas para 20 000 consultas × 60 configs (1,2 M instancias). Pero la cifra empírica de LLMRouterBench (≈ 1 000 horas A800 para 229 600 instancias, ≈ 15,7 s por instancia con modelos de razonamiento) extrapolada a 1,2 M instancias da ≈ 5 200 horas. **La verdad estará entre ambas cifras (≈ 150–5 000 GPU-h).** Recomendaciones:

1. Haz el piloto de 1 000 consultas y mide tok/s y tokens de salida reales antes de comprometerte.
2. Reduce el diseño antes de reducir el rigor: 8 000 consultas, 6–8 modelos y 3 niveles de esfuerzo, ≈ 25–30 configs.
3. Modelos de 70B y 235B necesitan 4–8 GPU (tensor parallel): son los que más encarecen; considera que entren solo en un subconjunto de consultas o por API alquilada de pesos abiertos.

## 5. Lo que NO he podido hacer (y qué implica)

- **No descargué ni ejecuté nada sobre datos reales**: el entorno no tiene red ni GPU. Todo el kit está probado con datos sintéticos y un simulador (`MockBackend`).
- **Esquemas de ficheros no verificados**: RouterBench (nombres de columnas), SPROUT, R2-Bench y los datos de LLMRouterBench. Los adaptadores son defensivos y traen `inspect`; verifica el mapeo en tu máquina.
- **Licencias**: no aparecieron en ninguna ficha que consulté. Compruébalas antes de redistribuir.
- **R2-Bench**: no verifiqué que los datos estén publicados.
- **AIQ** es una implementación propia aproximada: contrasta con el código oficial de RouterBench antes de comparar números con otros artículos.
- **Garantía de CRC**: la prueba sintética (36 ensayos, riesgo medio de test 0,1495 con α = 0,15) es coherente, pero la garantía formal exige monotonía del riesgo respecto al umbral; el kit mide su violación (`monotonicity_violation`) y debe reportarse.
- Un hallazgo de las pruebas sintéticas: un router kNN simple **no** superó al mejor modelo único (PerfGain negativo). No es evidencia sobre datos reales, pero ilustra por qué H1 exige baselines fuertes.

## 6. Riesgos de datos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Contaminación de benchmarks en los modelos | Consultas posteriores al corte, partición privada del 10 %, auditoría de memorización |
| Juez sesgado (autopreferencia, longitud) | Conjunto de ≥ 3 familias, jueces fuera del pool, auditoría humana de 500 ítems |
| Cumplimiento irregular de límites de longitud en modelos pequeños | Registrar cumplimiento; truncar a presupuesto como R2-Bench; el router lo aprende |
| «Esfuerzo» por budget forcing ≠ esfuerzo nativo de modelos propietarios | Declararlo como limitación y como definición operativa |
| Deriva de pesos abiertos ≠ deriva de APIs cerradas | Declararlo; complementar con eventos de precio y de pool |
| Ejecución de código generado | Solo en contenedor aislado sin red ni credenciales |
| Entrenar por error con RouterArena | `assert_trainable` lo impide en el código |
