"""Evaluación en RouterArena con el evaluador OFICIAL (RouteWorks/RouterArena, Apache-2.0).

El evaluador oficial solo sabe llamar a APIs comerciales; no sirve modelos de pesos abiertos
locales. Este módulo hace de puente:

1. Lee los prompts YA renderizados por el propio RouterArena (`dataset/router_data_10.json`,
   generado por su `scripts/process_datasets/prep_datasets.py`): no reimplementamos plantillas.
2. Genera las respuestas de cada candidato con nuestros backends (vLLM, servidor compatible
   con OpenAI o Mock) y las guarda en caché en el formato `generated_result` oficial.
3. Construye el fichero de predicciones de un router (decisiones query → modelo) más las
   entradas `for_optimality` que el evaluador usa para las métricas de optimalidad.
4. Registra los modelos locales en la tabla de precios oficial (regla `local:` del propio
   RouterArena, sin inventar precios) y ejecuta `llm_evaluation/run.py`.

REGLAS DE RouterArena (README oficial): es un conjunto SOLO de evaluación. No se entrena ni
ajusta nada con él, y el router debe decidir con el contenido de la consulta, no con el
dataset de origen ni con la plantilla de evaluación. `routedata.adapters.base.assert_trainable`
bloquea el primer caso; el segundo es responsabilidad del router que produce las decisiones.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .backends import Backend, GenRequest, GenResult

ARENA_REPO = "https://github.com/RouteWorks/RouterArena.git"
ARENA_COMMIT = "18383b483b56a7e1d60fa28d938e064fde57cc63"  # main el 2026-11-10
SPLIT_FILES = {"sub_10": "router_data_10.json", "full": "router_data.json"}
SPLIT_SIZES = {"sub_10": 809, "full": 8400}
PROVIDER = "self-hosted"


# ------------------------------------------------------------------ prompts oficiales
def load_arena_prompts(arena_dir: str | Path, split: str = "sub_10") -> list[dict[str, str]]:
    """[{global_index, prompt}] tal y como los renderiza RouterArena (prompt_formatted)."""
    path = Path(arena_dir) / "dataset" / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} no existe: ejecuta `python scripts/run_routerarena.py setup` (descarga y prepara el dataset)."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for e in rows:
        gi, prompt = e.get("global index"), e.get("prompt_formatted") or e.get("prompt")
        if gi and prompt:
            out.append({"global_index": str(gi), "prompt": str(prompt)})
    expected = SPLIT_SIZES.get(split)
    if expected is not None and len(out) != expected:
        raise ValueError(f"{split}: {len(out)} prompts, se esperaban {expected}")
    return out


# ------------------------------------------------------------------ formato generated_result
def to_generated_result(g: GenResult, model: str) -> dict[str, Any]:
    """GenResult → `generated_result` oficial. El evaluador solo puntúa y cobra filas con
    success=True y output_tokens > 0 (issue #135 de RouterArena); el resto cuenta como fallo."""
    ok = bool(g.text) and g.out_tokens > 0
    return {
        "generated_answer": g.text,
        "success": ok,
        "provider": PROVIDER,
        "model_used": model,
        "requested_model": model,
        "token_usage": {
            "input_tokens": int(g.in_tokens),
            "output_tokens": int(g.out_tokens),
            "total_tokens": int(g.in_tokens + g.out_tokens),
            "reasoning_tokens": int(g.reasoning_tokens),
            "cached_input_tokens": int(g.cached_tokens or 0),
        },
        "finish_reason": g.finish_reason,
        "latency_ms": float(g.latency_s) * 1000.0,
    }


def is_valid_generation(gr: Any) -> bool:
    """Misma regla que `llm_evaluation/run.py::is_valid_generation` del evaluador oficial."""
    if not isinstance(gr, dict) or gr.get("success") is not True:
        return False
    out = (gr.get("token_usage") or {}).get("output_tokens")
    return isinstance(out, int | float) and not isinstance(out, bool) and out > 0


# ------------------------------------------------------------------ caché de generaciones
def _cache_path(cache_dir: str | Path, model: str) -> Path:
    return Path(cache_dir) / (model.replace("/", "__") + ".jsonl")


def load_cache(cache_dir: str | Path, model: str) -> dict[str, dict[str, Any]]:
    """{global_index: generated_result} de un modelo (última línea gana)."""
    p = _cache_path(cache_dir, model)
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["global_index"]] = rec["generated_result"]
    return out


def generate_candidates(
    prompts: list[dict[str, str]],
    models: Iterable[str],
    backend_factory: Callable[[str], Backend],
    cache_dir: str | Path,
    max_tokens: int = 4096,
    enable_thinking: bool | None = False,
    temperature: float = 0.0,
    batch: int = 64,
    log: Callable[..., None] = print,
) -> dict[str, int]:
    """Genera (reanudable) la respuesta de cada modelo a cada prompt. Reintenta filas no válidas.
    Devuelve {modelo: nº de respuestas válidas en caché}."""
    os.makedirs(cache_dir, exist_ok=True)
    summary = {}
    for model in models:
        cached = load_cache(cache_dir, model)
        todo = [p for p in prompts if not is_valid_generation(cached.get(p["global_index"]))]
        log(f"[{model}] {len(prompts) - len(todo)} en caché, {len(todo)} por generar")
        if todo:
            backend = backend_factory(model)
            with open(_cache_path(cache_dir, model), "a", encoding="utf-8") as fh:
                for i in range(0, len(todo), batch):
                    chunk = todo[i : i + batch]
                    reqs = [
                        GenRequest(
                            messages=[{"role": "user", "content": p["prompt"]}],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            enable_thinking=enable_thinking,
                            meta={"query_id": p["global_index"]},
                        )
                        for p in chunk
                    ]
                    for p, g in zip(chunk, backend.generate(reqs), strict=True):
                        gr = to_generated_result(g, model)
                        rec = {"global_index": p["global_index"], "generated_result": gr}
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        cached[p["global_index"]] = gr
                    fh.flush()
                    log(f"[{model}] {min(i + batch, len(todo))}/{len(todo)}")
        summary[model] = sum(is_valid_generation(cached.get(p["global_index"])) for p in prompts)
    return summary


# ------------------------------------------------------------------ fichero de predicciones
def _row(gi: str, prompt: str, model: str, gr: dict | None, for_optimality: bool) -> dict[str, Any]:
    return {
        "global index": gi,
        "prompt": prompt,
        "prediction": model,
        "generated_result": gr,
        "cost": None,
        "accuracy": None,
        "for_optimality": for_optimality,
    }


def build_predictions(
    prompts: list[dict[str, str]],
    choices: dict[str, str],
    model_pool: list[str],
    cache_dir: str | Path,
    include_optimality: bool = True,
) -> list[dict[str, Any]]:
    """Predicciones en el formato oficial. `choices` = {global_index: modelo elegido}.
    Con include_optimality añade, por consulta, una entrada `for_optimality` por cada OTRO
    modelo del pool (como `router_inference/generate_prediction_file.py` en sub_10)."""
    missing = [p["global_index"] for p in prompts if p["global_index"] not in choices]
    if missing:
        raise ValueError(f"Faltan decisiones para {len(missing)} consultas (p. ej. {missing[:3]})")
    unknown = sorted({m for m in choices.values() if m not in model_pool})
    if unknown:
        raise ValueError(f"Modelos elegidos fuera del pool: {unknown}")
    caches = {m: load_cache(cache_dir, m) for m in model_pool}
    rows = []
    for p in prompts:
        gi, chosen = p["global_index"], choices[p["global_index"]]
        rows.append(_row(gi, p["prompt"], chosen, caches[chosen].get(gi), False))
    if include_optimality:
        for p in prompts:
            gi, chosen = p["global_index"], choices[p["global_index"]]
            for m in model_pool:
                if m != chosen:
                    rows.append(_row(gi, p["prompt"], m, caches[m].get(gi), True))
    return rows


def write_router_files(
    arena_dir: str | Path, router_name: str, model_pool: list[str], predictions: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """Escribe router_inference/config/<router>.json y predictions/<router>.json en el checkout."""
    root = Path(arena_dir) / "router_inference"
    cfg = root / "config" / f"{router_name}.json"
    pred = root / "predictions" / f"{router_name}.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    pred.parent.mkdir(parents=True, exist_ok=True)
    pipeline = {"router_name": router_name, "router_cls_name": "precomputed", "models": list(model_pool)}
    cfg.write_text(json.dumps({"pipeline_params": pipeline}, indent=2), encoding="utf-8")
    pred.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg, pred


# ------------------------------------------------------------------ registro de modelos y precios
def register_local_models(arena_dir: str | Path, models: dict[str, float], bits: int = 16) -> list[str]:
    """Añade los modelos que servimos nosotros a universal_model_names.py y a
    model_cost/model_profiles.yaml con la regla `local:` oficial (precio de mercado de un modelo
    abierto de ese tamaño; `models` = {nombre: parámetros activos en miles de millones}).
    Idempotente. Devuelve los modelos añadidos. Después hay que ejecutar sus scripts de precios
    (`scripts/pricing/local_price.py --write` y `build_model_cost.py`): ver `evaluate`."""
    root = Path(arena_dir)
    names_py = root / "universal_model_names.py"
    profiles = root / "model_cost" / "model_profiles.yaml"
    names_src = names_py.read_text(encoding="utf-8")
    prof_src = profiles.read_text(encoding="utf-8")
    added = []
    for name, active_b in models.items():
        if f'"{name}"' not in names_src:
            marker = "universal_names = ["
            if marker not in names_src:
                raise ValueError(f"No encuentro `{marker}` en {names_py}")
            names_src = names_src.replace(marker, f'{marker}\n    "{name}",', 1)
            added.append(name)
        if f'"{name}"' not in prof_src:
            prof_src = prof_src.rstrip("\n") + (
                f'\n  "{name}":\n'
                f"    price: {{input: 0.0, output: 0.0}}  # provisional: lo fija local_price.py --write\n"
                f"    local: {{active_params_b: {active_b:g}, bits: {bits}}}\n"
                f"    sync: false\n"
                f'    source: "local-model rule (model_cost/local_pricing.yaml); servido con vLLM por routing_data_kit"\n'
            )
            if name not in added:
                added.append(name)
    names_py.write_text(names_src, encoding="utf-8")
    profiles.write_text(prof_src, encoding="utf-8")
    return added


def _run(cmd: list[str], cwd: str | Path) -> None:
    print("$", " ".join(cmd), f"(cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def refresh_prices(arena_dir: str | Path, uv: str = "uv") -> None:
    _run([uv, "run", "python", "scripts/pricing/local_price.py", "--write"], arena_dir)
    _run([uv, "run", "python", "scripts/pricing/build_model_cost.py"], arena_dir)
    # falla si algún modelo local quedó con el precio provisional
    _run([uv, "run", "python", "scripts/pricing/local_price.py", "--check"], arena_dir)


def evaluate(
    arena_dir: str | Path, router_name: str, split: str = "sub_10", uv: str = "uv"
) -> dict[str, Any]:
    """Ejecuta el evaluador oficial y devuelve su metrics.json."""
    _run(
        [uv, "run", "python", "./router_inference/check_config_prediction_files.py", router_name, split],
        arena_dir,
    )
    _run([uv, "run", "python", "./llm_evaluation/run.py", router_name, split, "--force"], arena_dir)
    return dict(json.loads((Path(arena_dir) / "metrics.json").read_text(encoding="utf-8")))


def setup(arena_dir: str | Path, commit: str = ARENA_COMMIT, uv: str = "uv") -> None:
    """Clona RouterArena en el commit fijado, instala su entorno (uv) y prepara el dataset (red)."""
    arena_dir = Path(arena_dir)
    if not (arena_dir / ".git").exists():
        _run(["git", "clone", ARENA_REPO, str(arena_dir)], ".")
    _run(["git", "fetch", "origin", commit], arena_dir)
    _run(["git", "checkout", "--detach", commit], arena_dir)
    _run([uv, "sync"], arena_dir)
    _run([uv, "run", "python", "./scripts/process_datasets/prep_datasets.py"], arena_dir)
