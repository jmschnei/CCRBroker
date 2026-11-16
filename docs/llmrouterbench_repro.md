# Reproducción de routers publicados de LLMRouterBench (Fase 1) — EN CURSO

Rama: `feat/llmrouterbench-repro`. Estado al pausar (2026-11-13): embeddings precalculados
10.752 / 21.604; ningún router ejecutado todavía.

## Qué se reproduce

Código oficial de [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) (MIT) en
`811df08`, clonado en `external/LLMRouterBench` (fuera de git) con su propio `.venv`
(Python 3.11; versiones exactas en `outputs/embeddings/llmrb_env_freeze.txt`). Protocolo del
apéndice B del paper (arXiv 2601.07206): split 70/30 por prompt, semillas 42/999/2024/2025/3407,
embeddings `gte-Qwen2-7B-instruct`.

| Router | Setting | Paper (Tablas 11/12) | Hiperparámetros (apéndice B.3) |
|---|---|---|---|
| Avengers | rendimiento (20 modelos) | 71.94 | k-means k=64 |
| EmbedLLM | rendimiento | 71.24 | mf.py, batch 32768 |
| GraphRouter | rendimiento-coste (13) | PF 67.79 / BL 59.70 / CF 56.98 | 10 000 épocas |
| Avengers-Pro | rendimiento-coste | curva α (68.58 en α=1) | α∈[0,1] paso 0.01, k=64 |
| RouteLLM (MF) | rendimiento-coste | 67.67 | GPT-5 vs Qwen3-235B, dim 128, 100 épocas, umbral 0.5 |

No factibles sin CUDA (reentrenan un 7B con DeepSpeed en 8×A800): RouterDC, MODEL-SAT,
HybridLLM, FrugalGPT.

## Cómo retomar

```bash
cd own_broker && git checkout feat/llmrouterbench-repro

# 1) Terminar los embeddings (reanuda desde la caché SQLite; ~2-3 h en M2 Max, textos largos al final)
TOKENIZERS_PARALLELISM=false caffeinate -i external/LLMRouterBench/.venv/bin/python \
    scripts/embedding_server.py precompute --texts outputs/embeddings/texts.jsonl --batch-size 8

# 2) Routers, primero seed 42 y de uno en uno (el orquestador arranca el servidor de embeddings)
python3 scripts/run_llmrouterbench_baselines.py run --routers avengers --seeds 42
#    ... embedllm, graphrouter, avengers_pro, routellm

# 3) Comparación con el paper
python3 scripts/run_llmrouterbench_baselines.py summary
```

Si `external/LLMRouterBench` no existe: el orquestador lo clona en el commit fijado y enlaza
`results/bench` → `data/raw/llmrouterbench/extracted/bench-release`; el `.venv` se recrea con
`uv venv --python 3.11` + `requirements.txt` + `sentence-transformers torch torch_geometric wandb
bert_score beartype joblib matplotlib seaborn` y los pines `transformers<4.52`,
`sentence-transformers<5`, `huggingface_hub<1.0`, `datasets<4.0`.

Para ejecutarlo en el cluster Slurm (contenedor pyxis, GPU CUDA): `docs/slurm_llmrouterbench.md`.

## Partes del orquestador aún sin validar

Escritas a partir del código upstream, pendientes de la primera ejecución real:
- GraphRouter: rutas que espera `run_exp.py` (relativas a `baselines/GraphRouter/`).
- Avengers-Pro: extracción de la curva α de la salida de `run_weight_ablation`.
- RouteLLM: claves por dataset en la salida de `evaluate_mf` (`_dataset_avg`).

## Desviaciones a documentar en los resultados

- Embeddings truncados a 8192 tokens (upstream: 25 000) → afecta a 337 de 21 604 prompts (1,6 %).
- Versiones de librerías más recientes que los pines de upstream; CPU/MPS en vez de CUDA.
- Configs `*_42.json` no publicadas: reconstruidas con el apéndice B.3; lo no especificado
  (p. ej. β=9 de Avengers) se toma de los configs genéricos del repo.
- EmbedLLM y GraphRouter (upstream) eligen la mejor época con el test: se reporta así para
  comparar con el paper, y además el valor de la última época.
