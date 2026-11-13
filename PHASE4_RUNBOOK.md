# Runbook · Fase 4 (AgentRoute, E5 · RQ5 · H5)

El código está probado **solo con datos sintéticos, un simulador (`MockBackend`) y un servidor HTTP falso** (7 pruebas propias; 4 suites en total). **Nada se ha ejecutado contra vLLM ni contra un entorno agéntico real.** Los adaptadores a τ²-Bench, ScienceWorld, SWE-bench y QA multi-salto son **plantillas sin implementar**: no he podido verificar sus APIs desde este entorno.

## 0. Qué responde esta fase y qué necesita de las anteriores

**H5:** en tareas multi-turno, enrutar *por paso* con coste de cambio de modelo supera a enrutar *por tarea* y a enrutar por paso *sin* ese coste, cuando se contabilizan caché y contexto.

| Necesita de… | Detalle |
|---|---|
| Fase 2 / 3 | Pool de modelos con revisiones fijas y **precios derivados de throughput medido en lotes** (`routedata/pricing.py`). **No** derives precios de las ejecuciones secuenciales de agentes: un episodio paso a paso no satura la GPU y el precio saldría inflado. Usa `make-prices` sobre la tabla `configs` de la Fase 2 o 3 |
| ConfigRoute completo | **No.** Basta con el pool y los precios |
| Juez LLM | **No.** El éxito de tarea viene del propio entorno |
| GPU | **Sí, más complicada:** un servidor vLLM por modelo y episodios concurrentes (`--workers`) para que haya batching real |

Puede ir en paralelo a la Fase 3 si tienes GPU para ambas.

## 1. Cómo se contabiliza el coste (núcleo de H5)

En cada paso el servidor recibe **todo el contexto**. El kit modela dos modos (`generate/agentroute.py`):
- `cache_mode="none"`: cada paso factura el contexto completo → cambiar de modelo **no** cuesta extra.
- `cache_mode="prefix"`: con caché de prefijos solo se factura lo no cacheado, y **la caché es específica de cada modelo** → al cambiar de modelo hay que re-procesar el contexto en el destino. `retention="keep"`: la caché de cada modelo sobrevive a las llamadas de otros; `"evict"`: solo cuenta si el paso anterior fue del mismo modelo.

El coste se **recalcula offline** desde el log (tokens de contexto, de salida y modelo por paso): puedes cambiar precios, caché y retención sin re-ejecutar episodios. En el informe se reportan siempre ambos modos. Si el servidor expone `cached_tokens`, `analyze` contrasta el modelo de caché con lo reportado.

## 2. Políticas que se comparan
- **Fijas** (cada modelo), **mixtas** (`switch_k2`, `switch_k4`, `random`: dan datos contrafactuales y cotas).
- **escalation:** empieza con el barato y escala al caro tras un error de herramienta (baseline práctico).
- **task_router:** elige un modelo por tarea (kNN sobre la descripción inicial; éxito y coste de los episodios fijos).
- **step_router_nopenalty:** por paso, *ignora* el estado de la caché.
- **step_router_penalty:** por paso, con el **prefill real no cacheado** de cada candidato (el coste de cambiar aparece solo). Es la política de H5.

Etiquetas de paso (`stepdata`): **`mode=local`** (por defecto, barato): P(acción válida) con `n_samples` muestras de *un solo paso* desde cada estado; exige que el entorno marque los errores con una observación que empiece por `ERROR`. **`mode=rollout`** (caro): continúa con un modelo de referencia hasta el final; captura consecuencias a largo plazo, pero su coste posterior incluye el cambio al modelo de referencia (sesga a favor del de referencia). Empieza con `local`.
Selección de λ: en tareas de **validación** (25 % del train, no usadas para ajustar), el λ de menor coste con éxito ≥ mejor fijo − 0,02. Nunca en test.

## 3. Entorno: qué implementar
Contrato (`generate/envs/adapters_template.py`): `reset(seed)→obs`, `step(texto)→(obs, done)`, `success()→[0,1]`, `clone()`. Errores de herramienta/formato → observación que **empiece por `ERROR`**. Usa `ReplayCloneMixin` si el entorno es determinista dada (semilla, acciones).
Orden de viabilidad que recomiendo (**sin verificar**): (1) QA multi-salto con búsqueda sobre un corpus local (determinista y barato); (2) ScienceWorld (texto; no necesita simulador de usuario); (3) τ²-Bench (simulador de usuario con LLM: usa un modelo local con T=0 y semilla fija y verifica determinismo); (4) SWE-bench Verified (contenedores, trayectorias largas: solo un subconjunto pequeño y probablemente sin contrafactuales).
Parsing de acciones: el texto del modelo es la acción; tu adaptador lo parsea (JSON de herramienta, ReAct…). Si usas *tool calling* nativo de vLLM, comprueba que el modelo y el parser de herramientas lo soportan en tu versión.

## 4. Pre-chequeos (antes de recolectar nada a escala)
1. **Determinismo:** `AG.check_deterministic(env_factory, seed, acciones)` ≥ 0,95. Si no, los clones por reproducción y los contrafactuales **no son válidos**.
2. **Validez de formato:** ≥ 80 % de acciones válidas por modelo en ~20 episodios (si un modelo casi nunca produce acciones parseables, el problema es el prompt/parsers, no el routing).
3. **Margen para enrutar** (50 tareas, episodios fijos → `analyze`): mira `headroom`. Si el modelo más barato ya resuelve casi todo, o el oráculo por tarea no baja el coste respecto al mejor fijo (razón de coste ≥ 0,8), hay poco que ganar por tarea; el enrutado por paso aún puede ayudar, pero decídelo con datos.
4. **GPU y concurrencia:** mide con `--workers` realistas; el throughput agregado es lo que fija el precio.
5. **Modelo de caché:** si el servidor reporta `cached_tokens`, `analyze` da el error relativo del modelo; si es alto, ajusta `retention` o revisa que la caché de prefijos esté activa.

## 5. Paso a paso
Los modelos van de **más barato a más caro**.

    # 0) Servidores (uno por modelo; verifica flags en TU versión de vLLM; la caché de prefijos debe estar activa)
    vllm serve org/Modelo-pequeno --port 8001 --enable-prefix-caching     # ejemplo, SIN verificar
    #   crea endpoints.json (ver configs/agent_endpoints.example.json)

    # 1) Tareas y partición por tarea
    python -m generate.run_phase4 init --data-dir data/agent --env hotpot --n-tasks 200 --test-frac 0.4
    # 2) Precios desde la Fase 2/3 (throughput medido en lotes)
    python -m generate.run_phase4 make-prices --data-dir data/agent --configs data/drift/s00/configs [--tps tps.json]
    # 3) Episodios fijos (todas las tareas) y mixtos (train)
    python -m generate.run_phase4 collect --data-dir data/agent --backend openai --endpoints endpoints.json --models M1,M2,M3 --workers 8
    # 4) Etiquetas por estado (train)
    python -m generate.run_phase4 stepdata --data-dir data/agent --backend openai --endpoints endpoints.json --models M1,M2,M3 --reference-model M3 --n-samples 3 --max-trajectories 100
    # 5) Entrenar routers, elegir λ en validación y evaluar EN LÍNEA sobre test
    python -m generate.run_phase4 evaluate --data-dir data/agent --backend openai --endpoints endpoints.json --models M1,M2,M3
    # 6) Informe
    python -m generate.run_phase4 analyze --data-dir data/agent --models M1,M2,M3

Las órdenes son reanudables (saltan lo ya hecho). `--backend mock` sirve para ensayar la tubería; `python -m generate.run_phase4 simulate` lo hace de extremo a extremo con un entorno sintético.

## 6. Cómo leer `AGENT_REPORT.md`
- **Resumen por política:** éxito (IC bootstrap por tarea), pasos, cambios de modelo, coste y latencia **en ambos modos de caché**, y coste por éxito.
- **Headroom:** éxito y coste del mejor fijo frente al oráculo por tarea.
- **H5:** diferencias **pareadas por tarea** de `step_router_penalty` y `step_router_nopenalty` frente a `task_router` (Δéxito, Δcoste con IC, McNemar sobre éxito, Holm). H5 se sostiene solo si `step_router_penalty` gana en `cache_mode=prefix` con IC que excluya 0 y **no** pierde éxito. Si gana en `none` pero no en `prefix`, es exactamente el artefacto que H5 quiere evitar.
- Un episodio truncado por `max_steps` cuenta como fallo: documenta el tope.

## 7. Presupuesto
`python estimate_budget.py --agent` (valores por defecto: 200 tareas × 3 modelos × 15 pasos, contexto 6 000 tokens) da **≈ 0,5 GPU-h** para los episodios fijos y **≈ 3 GPU-h** para las etiquetas locales (≈ 6 GPU-h con rollouts), con supuestos **muy optimistas** (throughput agregado de 2 500 tok/s, concurrencia perfecta, sin latencia de herramientas). Con contextos largos, modelos con razonamiento y concurrencia limitada, multiplica por un factor grande y **mide con 20–50 tareas antes de escalar**. En SWE-bench el coste lo domina el entorno, no el modelo.

## 8. Puertas de paso
- Determinismo ≥ 0,95 (si usas contrafactuales o clones).
- Tasa de acciones válidas ≥ 80 % en todos los modelos del pool.
- Hay margen medible (headroom) o un motivo para enrutar por paso.
- Modelo de caché validado (si el servidor reporta `cached_tokens`).
- ≥ 100 tareas de test para sacar conclusiones; con menos, IC demasiado anchos.

## 9. Limitaciones y lo que NO verifiqué
1. `OpenAICompatBackend` solo se probó contra un servidor falso local; el campo `prompt_tokens_details.cached_tokens` puede no existir en tu versión.
2. Ningún adaptador de entorno real está implementado; los flags de vLLM del ejemplo no están verificados.
3. Las etiquetas `local` ignoran las consecuencias a largo plazo de una mala elección; `rollout` las captura pero sesga hacia el modelo de referencia. Los routers (Ridge/kNN) son referencias simples, no CCR.
4. Un solo entorno y pocas tareas dan baja potencia: no generalices a «agentes».
5. En el simulador, un router por paso que ignora la caché cambia más de modelo y se beneficia menos de la caché que una política fija, pero la política con penalización **no** superó a un router por tarea cuando el modelo mediano ya resolvía las tareas. Es una propiedad del simulador, no evidencia sobre H5.
6. El coste de GPU depende del precio efectivo (utilización, concurrencia): analiza sensibilidad a precios y a retención de caché.
7. El simulador de usuario (τ²-Bench) añade ruido y dependencia de otro modelo: declara cuál y con qué semilla.

## 10. Qué sigue incompleto en el kit
Fase 5 (AdvRoute: CLI y runbook), E7 (conformal: CLI), bootstrap por evento en E4, y adaptadores reales de entornos.
