"""Fase 1 · Reproducción de 5 routers publicados con el código OFICIAL de LLMRouterBench.

Upstream: github.com/ynulihao/LLMRouterBench (MIT), fijado en LLMRB_COMMIT, clonado en
external/LLMRouterBench con su propio entorno (.venv). Protocolo del apéndice B del paper
(arXiv 2601.07206): split 70/30 por prompt, semillas 42/999/2024/2025/3407, embeddings
gte-Qwen2-7B-instruct para todos los métodos.

Routers (los factibles sin GPU CUDA; RouterDC, MODEL-SAT, HybridLLM y FrugalGPT reentrenan
un modelo de 7B con DeepSpeed en 8×A800):

  avengers      rendimiento (20 modelos ligeros)   k-means k=64 (Tabla 11: 71.94)
  embedllm      rendimiento                         mf.py, batch 32768 (Tabla 11: 71.24)
  graphrouter   rendimiento-coste (13 modelos)      PF/BL/CF, 10 000 épocas (Tabla 12: PF 67.79)
  avengers_pro  rendimiento-coste                   barrido α∈[0,1] paso 0.01 (Tabla 12)
  routellm      rendimiento-coste                   MF, GPT-5 vs Qwen3-235B, dim 128, 100 épocas (Tabla 12: 67.67)

Los embeddings los sirve scripts/embedding_server.py (API compatible OpenAI) en local; el
orquestador lo arranca si no está levantado. Uso:

    python scripts/run_llmrouterbench_baselines.py run --routers avengers,embedllm --seeds 42
    python scripts/run_llmrouterbench_baselines.py summary
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LLMRB = ROOT / "external" / "LLMRouterBench"
LLMRB_REPO = "https://github.com/ynulihao/LLMRouterBench.git"
LLMRB_COMMIT = "811df08b1b37646064eeb8cb6acfd498bf99885a"
PY = LLMRB / ".venv" / "bin" / "python"
OUT = ROOT / "outputs" / "llmrouterbench_repro"
SEEDS = (42, 999, 2024, 2025, 3407)
SPLIT = 0.7
PORT = int(os.environ.get("LLMRB_EMB_PORT", "8765"))  # en Slurm: un puerto por trabajo
EMB_URL = f"http://127.0.0.1:{PORT}/v1"
EMB_MODEL = "gte_Qwen2-7B-instruct"
PERF_CFG = "config/baseline_config.yaml"
COST_CFG = "config/baseline_config_performance_cost.yaml"
STRONG, WEAK = "gpt-5", "qwen3-235b-a22b-2507"  # apéndice B.2 (iii)

# Valores publicados (Tablas 11 y 12, media sobre los datasets del setting, test 30 %)
PAPER = {
    "avengers": 71.94,
    "embedllm": 71.24,
    "graphrouter_PF": 67.79,
    "graphrouter_BL": 59.70,
    "graphrouter_CF": 56.98,
    "routellm": 67.67,
    "avengers_pro_alpha1.00": 68.58,
}

# Reaprovechar salidas de una ejecución anterior completa (no reentrena): --reuse o LLMRB_REUSE=1
REUSE = os.environ.get("LLMRB_REUSE", "") not in ("", "0")

ENV = {**os.environ, "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}


# ------------------------------------------------------------------ infraestructura
def _run(cmd: list[str], log: Path, cwd: Path = LLMRB) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(map(str, cmd)), f"  > {log.relative_to(ROOT)}", flush=True)
    with open(log, "w") as fh:
        p = subprocess.run(list(map(str, cmd)), cwd=cwd, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    text = log.read_text(errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"Falló ({p.returncode}): {' '.join(map(str, cmd))}\n{text[-3000:]}")
    return text


def ensure_checkout() -> None:
    if not (LLMRB / ".git").exists():
        subprocess.run(["git", "clone", LLMRB_REPO, str(LLMRB)], check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=LLMRB, capture_output=True, text=True
    ).stdout.strip()
    if head != LLMRB_COMMIT:  # evita carreras en el índice de git con varios trabajos Slurm a la vez
        subprocess.run(["git", "checkout", "-q", "--detach", LLMRB_COMMIT], cwd=LLMRB, check=True)
    bench = LLMRB / "results" / "bench"
    if not bench.exists():
        bench.symlink_to(ROOT / "data" / "raw" / "llmrouterbench" / "extracted" / "bench-release")


def embedding_config() -> Path:
    """Config de embeddings de upstream apuntando al servidor local (sin caché MySQL)."""
    path = OUT / "embedding_config_local.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "embedding_model:\n"
        f"  name: {EMB_MODEL}\n"
        f"  api_model_name: {EMB_MODEL}\n"
        "  generator_type: embedding\n"
        f"  base_url: {EMB_URL}\n"
        "  api_key: local-key\n"
        "  timeout: 600\n"
        "  max_context_length: 25000\n"
        "cache:\n"
        "  enabled: false\n"
    )
    return path


def ensure_embedding_server() -> subprocess.Popen | None:
    def up() -> bool:
        try:
            with urllib.request.urlopen(f"{EMB_URL}/models", timeout=2):
                return True
        except OSError:
            return False

    if up():
        return None
    log = OUT / "embedding_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [str(PY), str(ROOT / "scripts" / "embedding_server.py"), "serve", "--port", str(PORT)],
        stdout=open(log, "a"),
        stderr=subprocess.STDOUT,
        env=ENV,
    )
    for _ in range(60):
        if up():
            return proc
        time.sleep(1)
    raise RuntimeError(f"El servidor de embeddings no arrancó; ver {log}")


def _save(router: str, seed: int, result: dict[str, Any]) -> Path:
    path = OUT / router / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"router": router, "seed": seed, "llmrb_commit": LLMRB_COMMIT, **result}, indent=2)
    )
    print(f"→ {path.relative_to(ROOT)}: {result.get('accuracy')}", flush=True)
    return path


def _floats(pattern: str, text: str) -> list[float]:
    return [float(x) for x in re.findall(pattern, text)]


# ------------------------------------------------------------------ routers
def run_avengers(seed: int, emb: Path) -> None:
    d = f"baselines/AvengersPro/data/small_models/seed{seed}_split{SPLIT}"
    work = OUT / "avengers" / f"seed{seed}"
    out = work / "results.json"
    if REUSE and out.exists() and (work / "config.json").exists():
        print(f"(reuse) {out.relative_to(ROOT)}", flush=True)
        _save(
            "avengers",
            seed,
            {**_avengers_metrics(out), "config": json.loads((work / "config.json").read_text())},
        )
        return
    _run(
        [
            PY,
            "-m",
            "baselines.adaptors.avengerspro_adaptor",
            "--config",
            PERF_CFG,
            "--seed",
            seed,
            "--split-ratio",
            SPLIT,
            "--output-dir",
            "baselines/AvengersPro/data/small_models",
        ],
        work / "adaptor.log",
    )
    cfg = {
        "train_data_path": f"{d}/train.jsonl",
        "test_data_path": f"{d}/test.jsonl",
        "baseline_scores_path": f"{d}/baseline_scores.json",
        "n_clusters": 64,  # apéndice B.3
        "seed": seed,
        "max_router": 1,
        "top_k": 1,
        "beta": 9.0,
        "max_workers": 8,
        "cluster_batch_size": 1000,
        "embedding_model": EMB_MODEL,
        "embedding_base_url": EMB_URL,
        "embedding_api_key": "local-key",
        "embedding_config_path": str(emb),
        "excluded_models": [],
        "ood_datasets": [],
    }
    (work / "config.json").write_text(json.dumps(cfg, indent=2))
    _run(
        [
            PY,
            "-m",
            "baselines.AvengersPro.simple_cluster_router",
            "--config",
            work / "config.json",
            "--output",
            out,
        ],
        work / "router.log",
    )
    _save("avengers", seed, {**_avengers_metrics(out), "config": cfg})


def _avengers_metrics(out: Path) -> dict[str, Any]:
    res = json.loads(out.read_text())["results"]  # {timestamp, config, results}
    return {
        "accuracy": res["accuracy"],  # media por dataset, en %
        "sample_avg": res.get("all_sample_avg"),
        "total_cost": res.get("cost_analysis", {}).get("total_cost"),
        "per_dataset": {k: v.get("accuracy") for k, v in res.get("dataset_performance", {}).items()},
    }


def run_avengers_pro(seed: int, emb: Path) -> None:
    d = f"baselines/AvengersPro/data/proprietary_models/seed{seed}_split{SPLIT}"
    work = OUT / "avengers_pro" / f"seed{seed}"
    _run(
        [
            PY,
            "-m",
            "baselines.adaptors.avengerspro_adaptor",
            "--config",
            COST_CFG,
            "--seed",
            seed,
            "--split-ratio",
            SPLIT,
            "--output-dir",
            "baselines/AvengersPro/data/proprietary_models",
        ],
        work / "adaptor.log",
    )
    base = json.loads((LLMRB / "baselines/AvengersPro/config/balance_config.json").read_text())
    base.update(
        {
            "train_data_path": f"{d}/train.jsonl",
            "test_data_path": f"{d}/test.jsonl",
            "baseline_scores_path": f"{d}/baseline_scores.json",
            "n_clusters": 64,  # apéndice B.3
            "seed": seed,
            "max_workers": 8,
            "embedding_model": EMB_MODEL,
            "embedding_base_url": EMB_URL,
            "embedding_api_key": "local-key",
            "embedding_config_path": str(emb),
        }
    )
    (work / "config.json").write_text(json.dumps(base, indent=2))
    _run(
        [
            PY,
            "-m",
            "baselines.AvengersPro.ablation.run_weight_ablation",
            "--config",
            work / "config.json",
            "--weight-range",
            "0,1",
            "--step-size",
            "0.01",
            "--output",
            work / "ablation",
            "--no-visualizations",
            "--log-level",
            "WARNING",
        ],
        work / "ablation.log",
    )
    concise = json.loads((work / "ablation" / "results" / "weight_ablation_concise.json").read_text())
    curve = [
        {
            "performance_weight": round(float(r["performance_weight"]), 4),
            "accuracy": float(r["non_ood_accuracy"]),  # media por dataset, en %
            "sample_avg": float(r["non_ood_sample_avg"]),
            "total_cost": float(r["non_ood_total_cost"]),
        }
        for r in concise["router_results"]
    ]
    a1 = next((p for p in curve if abs(p["performance_weight"] - 1.0) < 1e-9), None)
    _save("avengers_pro", seed, {"accuracy": a1["accuracy"] if a1 else None, "curve": curve, "config": base})


def run_embedllm(seed: int, emb: Path) -> None:
    base = "baselines/EmbedLLM/data/small_models"
    d = f"{base}/seed{seed}_split{SPLIT}"
    work = OUT / "embedllm" / f"seed{seed}"
    train_log = work / "train.log"
    if REUSE and train_log.exists() and "BEST PERFORMANCE" in train_log.read_text(errors="replace"):
        print(f"(reuse) {train_log.relative_to(ROOT)}", flush=True)
        _save("embedllm", seed, _embedllm_metrics(train_log.read_text(errors="replace")))
        return
    _run(
        [
            PY,
            "-m",
            "baselines.adaptors.embedllm_adaptor",
            "--config",
            PERF_CFG,
            "--seed",
            seed,
            "--split-ratio",
            SPLIT,
            "--output-dir",
            base,
            "--embedding-config",
            emb,
        ],
        work / "adaptor.log",
    )
    log = _run(
        [
            PY,
            "mf.py",
            "--train-data-path",
            f"../../../{d}/train_ours.csv",
            "--test-data-path",
            f"../../../{d}/test_ours.csv",
            "--question-embedding-path",
            f"../../../{d}/question_embeddings.pth",
            "--eval-mode",
            "router",
            "--model-num",
            20,
            "--batch-size",
            32768,  # apéndice B.3
            "--model-save-path",
            work.resolve() / "model.pth",
            "--embedding-save-path",
            work.resolve() / "model_embeddings.pth",
        ],
        work / "train.log",
        cwd=LLMRB / "baselines" / "EmbedLLM" / "algorithm",
    )
    _save("embedllm", seed, _embedllm_metrics(log))


def _embedllm_metrics(log: str) -> dict[str, Any]:
    """Métricas de test de mf.py (modo router).

    Por época, mf.py imprime en el bloque de TEST una línea sin sangría
    "Dataset-Level Average Accuracy: X". El evaluador imprime además, con sangría y el sufijo
    "(unweighted)", la misma métrica para train y test: esas se excluyen. Al final imprime
    "Best Dataset-Level Accuracy: X at Epoch N" (criterio upstream: mejor época según el test).
    tqdm escribe en la misma salida con \\r, así que no se ancla al inicio de línea.
    """
    per_epoch = _floats(r"(?m)Dataset-Level Average Accuracy: ([0-9.]+)[ \t]*$", log)
    best = re.findall(r"Best Dataset-Level Accuracy: ([0-9.]+) at Epoch (\d+)", log)
    if not per_epoch and not best:
        raise RuntimeError("No se encontraron métricas de test en el log de EmbedLLM (mf.py)")
    best_acc = float(best[-1][0]) if best else max(per_epoch)
    return {
        "accuracy": 100 * best_acc,  # criterio upstream: mejor época según test
        "best_epoch": int(best[-1][1]) if best else None,
        "accuracy_last_epoch": 100 * per_epoch[-1] if per_epoch else None,
        "epochs_evaluated": len(per_epoch),
    }


def run_graphrouter(seed: int, emb: Path, scenarios=("PF", "BL", "CF")) -> None:
    names = {"PF": "Performance First", "BL": "Balance", "CF": "Cost First"}
    out_dir = f"baselines/GraphRouter/data/seed_{seed}_{SPLIT}"
    work = OUT / "graphrouter" / f"seed{seed}"
    _run(
        [
            PY,
            "-m",
            "baselines.adaptors.graphrouter_adaptor",
            "--baseline-config",
            COST_CFG,
            "--graphrouter-config",
            "baselines/GraphRouter/configs/adaptor_config.yaml",
            "--embedding-config",
            emb,
            "--output-dir",
            out_dir,
            "--seed",
            seed,
        ],
        work / "adaptor.log",
    )
    base_cfg = (LLMRB / "baselines/GraphRouter/configs/config.yaml").read_text()
    import yaml  # disponible en nuestro entorno

    cfg = yaml.safe_load(base_cfg)
    results = {}
    for sc in scenarios:
        data = LLMRB / out_dir
        cfg.update(
            {
                "llm_description_path": str(data / "LLM_Descriptions.json"),
                "llm_embedding_path": str(data / "llm_description_embedding.pkl"),
                "saved_router_data_path": str(data / "router_data.csv"),
                "model_path": str((work / f"best_{sc}.pth").resolve()),
                "train_epoch": 10000,  # apéndice B.3
                "scenario": names[sc],
                "seed": seed,
                "split_ratio": [SPLIT, 0.0, round(1 - SPLIT, 4)],
            }
        )
        cfg_path = work / f"config_{sc}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        log = _run(
            [PY, "run_exp.py", "--config_file", cfg_path.resolve()],
            work / f"train_{sc}.log",
            cwd=LLMRB / "baselines" / "GraphRouter",
        )
        # graph_nn.py imprime primero el "BEST TEST CHECKPOINT" (criterio upstream) y después "LAST EPOCH"
        acc = _floats(r"Dataset-Level Average Accuracy:\s+([0-9.]+)", log)
        cost = _floats(r"(?m)^\s*Total Cost:\s+([0-9.]+)", log)
        results[sc] = {
            "accuracy": 100 * acc[0] if acc else None,
            "accuracy_last_epoch": 100 * acc[-1] if len(acc) > 1 else None,
            "total_cost": cost[0] if cost else None,
            "total_cost_last_epoch": cost[-1] if len(cost) > 1 else None,
        }
    _save("graphrouter", seed, {"accuracy": results.get("PF", {}).get("accuracy"), "scenarios": results})


def run_routellm(seed: int, emb: Path) -> None:
    pair = f"{STRONG}__vs__{WEAK}"
    data_dir = LLMRB / "baselines/RouteLLM/data" / f"seed{seed}_split{SPLIT}_{pair}"
    work = OUT / "routellm" / f"seed{seed}"
    _run(
        [
            PY,
            "-m",
            "baselines.adaptors.routellm_adaptor",
            "--config",
            COST_CFG,
            "--strong-model",
            STRONG,
            "--weak-model",
            WEAK,
            "--seed",
            seed,
            "--split-ratio",
            SPLIT,
            "--output-dir",
            "baselines/RouteLLM/data",
            "--embedding-config",
            emb,
        ],
        work / "adaptor.log",
    )
    ckpt = (work / "checkpoint").resolve()
    train_cfg = {
        "json_path": str(data_dir / "pairwise_train.json"),
        "npy_path": str(data_dir / "prompt_embeddings.npy"),
        "dim": 128,  # apéndice B.3
        "use_proj": True,
        "batch_size": 64,
        "num_epochs": 100,  # apéndice B.3
        "alpha": 0.1,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "device": "cpu",
        "save_path": str(ckpt / "mf_model.pt"),
    }
    (work / "train_config.json").write_text(json.dumps(train_cfg, indent=2))
    _run(
        [
            PY,
            "-m",
            "baselines.RouteLLM.routers.matrix_factorization.train_matrix_factorization",
            "--config",
            work / "train_config.json",
        ],
        work / "train.log",
    )
    eval_cfg = json.loads((LLMRB / "baselines/RouteLLM/router_eval_config.json").read_text())
    eval_cfg["mf"].update(
        {
            "checkpoint_path": str(ckpt),
            "hidden_size": 128,
            "use_proj": True,
            "embedding_config_path": str(emb),
        }
    )
    (work / "eval_config.json").write_text(json.dumps(eval_cfg, indent=2))
    out = work / "eval.json"
    _run(
        [
            PY,
            "-m",
            "baselines.RouteLLM.evaluate_mf",
            "--config",
            work / "eval_config.json",
            "--data-dir",
            data_dir,
            "--strong-model",
            STRONG,
            "--weak-model",
            WEAK,
            "--threshold",
            0.5,
            "--output",
            out,
        ],
        work / "eval.log",
    )
    res = json.loads(out.read_text())
    _save("routellm", seed, {"accuracy": _dataset_avg(res), "raw": res, "train_config": train_cfg})


def _dataset_avg(res: dict[str, Any]) -> float | None:
    """Media por dataset de la accuracy del modelo elegido (definición de las Tablas 11/12), en %."""
    v = res.get("all_dataset_avg")
    return 100 * v if v is not None else None


TEXTS_CODE = """
import json, sys
from baselines import BaselineDataLoader
texts = set()
for cfg in ["config/baseline_config.yaml", "config/baseline_config_performance_cost.yaml"]:
    texts |= {r.prompt for r in BaselineDataLoader(cfg).load_all_records()}
with open(sys.argv[1], "w") as f:
    for t in sorted(texts):
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\\n")
print(len(texts), "prompts únicos")
"""


def export_texts(dest: Path) -> None:
    """Lista de prompts únicos (ambos settings) que embeben los adaptadores: entrada de `precompute`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run([PY, "-c", TEXTS_CODE, dest.resolve()], OUT / "export_texts.log")
    print(f"→ {dest}", flush=True)


ROUTERS = {
    "avengers": run_avengers,
    "embedllm": run_embedllm,
    "graphrouter": run_graphrouter,
    "avengers_pro": run_avengers_pro,
    "routellm": run_routellm,
}


# ------------------------------------------------------------------ resumen
def summary() -> dict[str, Any]:
    import statistics as st

    rows = {}
    for router in ROUTERS:
        files = sorted((OUT / router).glob("seed*.json"))
        accs = [json.loads(f.read_text()).get("accuracy") for f in files]
        accs = [a for a in accs if a is not None]
        if not accs:
            continue
        ref = PAPER.get(router) or PAPER.get(f"{router}_PF") or PAPER.get(f"{router}_alpha1.00")
        rows[router] = {
            "seeds": len(accs),
            "mean": st.mean(accs),
            "std": st.stdev(accs) if len(accs) > 1 else 0.0,
            "paper": ref,
            "diff": (st.mean(accs) - ref) if ref is not None else None,
        }
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"{'router':14} {'seeds':>5} {'ours':>14} {'paper':>7} {'diff':>7}")
    for r, v in rows.items():
        print(
            f"{r:14} {v['seeds']:>5} {v['mean']:7.2f} ± {v['std']:4.2f} {v['paper'] or float('nan'):7.2f} "
            f"{(v['diff'] if v['diff'] is not None else float('nan')):+7.2f}"
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("setup", "texts", "run", "summary"))
    ap.add_argument("--texts-out", type=Path, default=ROOT / "outputs" / "embeddings" / "texts.jsonl")
    ap.add_argument("--routers", default=",".join(ROUTERS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="reaprovecha logs/resultados de ejecuciones completas (Avengers, EmbedLLM) sin reentrenar",
    )
    a = ap.parse_args(argv)
    if a.reuse:
        global REUSE
        REUSE = True
    if a.cmd == "summary":
        summary()
        return
    ensure_checkout()
    if a.cmd == "setup":
        return
    if a.cmd == "texts":
        export_texts(a.texts_out)
        return
    emb = embedding_config()
    server = ensure_embedding_server()
    try:
        for seed in [int(s) for s in a.seeds.split(",")]:
            for r in a.routers.split(","):
                ROUTERS[r.strip()](seed, emb)
    finally:
        if server is not None:
            server.terminate()
    summary()


if __name__ == "__main__":
    sys.exit(main())
