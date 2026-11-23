# Reproducción de LLMRouterBench en el cluster Slurm

Ejecuta en el cluster lo mismo que en el Mac (`docs/llmrouterbench_repro.md`): los embeddings
gte-Qwen2-7B de los 21 604 prompts y los 5 routers publicados (Avengers, EmbedLLM, GraphRouter,
Avengers-Pro y RouteLLM) × 5 semillas. Los trabajos corren en contenedor (pyxis/enroot) con
estos montajes:

```
/netscratch/$USER:/netscratch/$USER, /ds:/ds:ro, <repo>:<repo>
```

## Ficheros

| Fichero | Qué hace |
|---|---|
| `slurm/llmrb/config.sh` | Variables: ruta del repo, imagen, particiones, truncado, cachés en netscratch |
| `slurm/llmrb/submit.sh` | Lo que se ejecuta en el nodo de login: envía cada paso con sus recursos y dependencias |
| `slurm/llmrb/job.sbatch` | Envoltorio: `srun --container-image … --container-mounts …` → `inner.sh` |
| `slurm/llmrb/inner.sh` | Pasos dentro del contenedor: `setup`, `precompute`, `routers` |
| `slurm/llmrb/requirements.lock.txt` | Versiones exactas del entorno de upstream (las mismas del Mac) |

| Paso | Recursos (por defecto) | Qué hace | Duración estimada |
|---|---|---|---|
| `setup` | CPU, 8 cores, 32 GB, 3 h | Instala uv; clona LLMRouterBench en `811df08`; descarga y extrae `bench-release` (1,2 GB → 6,5 GB); crea `.venv` (Python 3.11); descarga el modelo (15 GB); genera `texts.jsonl` | 30–60 min |
| `precompute` | 1 GPU, 64 GB, 6 h | Rellena la caché SQLite de embeddings (10,3 M tokens) | < 1 h en A100 |
| `routers <seed>` | 1 GPU, 16 cores, 128 GB, 48 h | Los 5 routers para una semilla | sin medir todavía (GraphRouter: 3 × 10 000 épocas) |

Todo lo que se descarga o genera se queda en `/netscratch/$USER`: el repo, `$HF_HOME`, la caché
de uv y su Python 3.11, y `outputs/`. En `$HOME` no se escribe nada.

## 0. Antes de empezar (una vez)

En el nodo de login, comprueba qué imagen y particiones hay y, si no coinciden con los valores por
defecto, ajústalos en `slurm/llmrb/config.sh` o pásalos por el entorno:

```bash
ls /enroot/ | grep -i pytorch        # por defecto: /enroot/nvcr.io_nvidia_pytorch_24.12-py3.sqsh
sinfo -s                             # por defecto: GPU=A100-80GB, CPU=batch
```

Basta cualquier imagen que tenga `bash`, `git`, `curl`, `python3` y el driver de NVIDIA, porque
PyTorch se instala en el `.venv` propio y no se usa el de la imagen. Si el driver del cluster es
antiguo para la build CUDA por defecto de `torch==2.15.0`, define `TORCH_INDEX_URL` (ver
*Problemas frecuentes*).

## 1. Código en el cluster

La rama tiene que estar en GitHub. Desde el Mac:

```bash
git push -u origin feat/llmrouterbench-repro
```

En el cluster:

```bash
cd /netscratch/$USER
git clone https://github.com/jmschnei/CCRBroker.git own_broker
cd own_broker
git checkout feat/llmrouterbench-repro
```

Si el repo está en otra carpeta, exporta `REPO_DIR=/netscratch/$USER/<carpeta>` antes de
ejecutar `submit.sh`. Si no lo haces, `submit.sh` toma la carpeta en la que está.

### (Opcional) Reaprovechar los embeddings ya calculados en el Mac

Solo sirve con el truncado por defecto (`EMB_MAX_SEQ=8192`), y te ahorra el paso `precompute`.
Desde el Mac:

```bash
rsync -avP outputs/embeddings/gte_qwen2_7b.sqlite outputs/embeddings/texts.jsonl \
    <usuario>@<login-cluster>:/netscratch/<usuario>/own_broker/outputs/embeddings/
```

Para ahorrar también la descarga de 1,2 GB, puedes copiar el tarball:
`data/raw/llmrouterbench/bench-release.tar.gz` → la misma ruta en el cluster.

## 2. Ejecutar

Todos los comandos se lanzan desde la raíz del repo en el nodo de login.

**Opción A: todo encadenado** (setup → precompute → routers con seed 42):

```bash
./slurm/llmrb/submit.sh pipeline
```

**Opción B: paso a paso**, recomendada la primera vez:

```bash
./slurm/llmrb/submit.sh setup
# cuando termine OK:
./slurm/llmrb/submit.sh precompute
# cuando termine OK (debe imprimir "embeddings en caché: 21604"):
./slurm/llmrb/submit.sh routers 42
```

Para validar los routers de uno en uno (el orquestador tiene partes que todavía no se han probado
con datos reales; ver `docs/llmrouterbench_repro.md`):

```bash
ROUTERS=avengers ./slurm/llmrb/submit.sh routers 42
ROUTERS=embedllm ./slurm/llmrb/submit.sh routers 42
ROUTERS=graphrouter,avengers_pro,routellm ./slurm/llmrb/submit.sh routers 42
```

**Resto de semillas.** Cuando seed 42 cuadre con el paper:

```bash
./slurm/llmrb/submit.sh routers all          # array 0-4 → semillas 42 999 2024 2025 3407
# o solo las que faltan:
for s in 999 2024 2025 3407; do ./slurm/llmrb/submit.sh routers $s; done
```

Cada trabajo de router copia la caché de embeddings a disco local (`$TMPDIR`) y arranca su propio
servidor de embeddings en un puerto distinto. Así varias semillas pueden correr a la vez, incluso
en el mismo nodo. El modelo de 7B solo se carga en GPU si falta algún texto en la caché; en la
práctica pasa solo con las descripciones de modelos y tareas que embebe GraphRouter.

**Embeddings sin truncado propio** (alternativa a la desviación de 8192 tokens): se escribe en
otra caché, `gte_qwen2_7b_L32768.sqlite`, y hay que usarla de forma coherente en `precompute` y
en `routers`:

```bash
EMB_MAX_SEQ=32768 EMB_BATCH=4 ./slurm/llmrb/submit.sh precompute
EMB_MAX_SEQ=32768 ./slurm/llmrb/submit.sh routers 42
```

Los resultados se escriben en el mismo `outputs/llmrouterbench_repro/`. Si quieres conservar las
dos variantes, mueve esa carpeta entre una ejecución y otra.

## 3. Seguimiento

```bash
squeue --me
tail -f outputs/slurm/llmrb-emb-<jobid>_*.out                       # progreso de embeddings
tail -f outputs/slurm/llmrb-routers-<jobid>_*.out                   # orquestador
ls outputs/llmrouterbench_repro/*/seed42/                           # logs por router (adaptor/train/eval)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel <jobid>
```

## 4. Resultados

No hace falta contenedor (solo usa la biblioteca estándar):

```bash
python3 scripts/run_llmrouterbench_baselines.py summary   # tabla nuestra vs. paper
```

Para traer los resultados al Mac:

```bash
rsync -avP <usuario>@<login-cluster>:/netscratch/<usuario>/own_broker/outputs/llmrouterbench_repro/ \
    outputs/llmrouterbench_repro_cluster/
```

Archivos que conviene guardar con los resultados: `outputs/slurm/llmrb_env_freeze_cluster.txt`
(entorno real del cluster) y el log de `precompute` (GPU y versión de CUDA).

## Problemas frecuentes

- **`sin CUDA` en `precompute`, o `torch` sin GPU:** la build de PyTorch del lock no coincide con
  el driver (`nvidia-smi` muestra la versión de CUDA soportada). Reinstala torch desde el índice
  adecuado y relanza `setup` (es idempotente):
  `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 TORCH_VERSION=<versión con esa build> ./slurm/llmrb/submit.sh setup`.
- **OOM en `precompute`:** el script reduce el batch a la mitad automáticamente y reintenta. Si
  sigue fallando, prueba con `EMB_BATCH=4` o una GPU con más memoria.
- **El trabajo de `setup` falla en la descarga:** el nodo de cómputo no tiene salida a internet.
  Ejecuta los pasos de descarga de `inner.sh setup` (uv, `git clone`, tarball, `uv pip install`,
  `snapshot_download`) en el nodo de login dentro de `srun --container-image …` o copia los
  artefactos con rsync.
- **`El servidor de embeddings no arrancó`:** mira `outputs/llmrouterbench_repro/embedding_server.log`.
  Si el puerto está ocupado, exporta `LLMRB_EMB_PORT` con otro valor.
- **Un router terminó pero falló al leer su resultado:** con `LLMRB_REUSE=1` (o `--reuse`) se
  vuelven a leer los logs y resultados ya completos de Avengers y EmbedLLM sin reentrenar, y la
  ejecución sigue con los demás routers:
  `LLMRB_REUSE=1 ./slurm/llmrb/submit.sh routers 42`.
- **Volver a lanzar un router:** se sobrescriben `outputs/llmrouterbench_repro/<router>/seed<k>*`.
  Los adaptadores regeneran sus datos en `external/LLMRouterBench/baselines/*/data/`.

## Desviaciones respecto al paper

Son las mismas que en `docs/llmrouterbench_repro.md`, salvo dos diferencias:

- En CUDA se puede eliminar el truncado a 8192 tokens con `EMB_MAX_SEQ=32768`.
- Los embeddings se calculan en fp16, como en el Mac. Entre GPUs pueden diferir en el último
  decimal, lo que puede cambiar alguna asignación de k-means.
