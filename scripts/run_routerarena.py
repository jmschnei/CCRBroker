"""Fase 1 · Evaluación en RouterArena (sub_10) con el evaluador oficial.

Flujo (ver RUNBOOK_COMPLETE.md, Fase 1, paso 5):

    # 1) una vez, con red (sin GPU): clona RouterArena en el commit fijado, uv sync, dataset
    python scripts/run_routerarena.py setup

    # 2) GPU: respuestas de cada candidato del pool a los 809 prompts oficiales (reanudable)
    python scripts/run_routerarena.py generate --backend vllm [--models m1,m2]
    #    o contra servidores `vllm serve`:  --backend openai --endpoints configs/agent_endpoints.json

    # 3) CPU: cada candidato como "router" de un solo modelo (accuracy/coste por modelo)
    python scripts/run_routerarena.py candidates
    #    (caché y resultados van a outputs/routerarena/<backend>/; usa el mismo --backend en todos los pasos)

    # 4) CPU: un router propio a partir de sus decisiones {global_index: modelo}
    python scripts/run_routerarena.py evaluate --router mi-router --choices decisiones.json

Las decisiones deben salir de un router que NO se ha entrenado ni ajustado con RouterArena y
que decide con el contenido de la consulta (reglas del README oficial).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate import routerarena as RA
from generate.backends import MockBackend, OpenAICompatBackend

DEFAULTS = {
    "arena_dir": ROOT / "external" / "RouterArena",
    "pool": ROOT / "configs" / "pool_openweights.yaml",
    # caché y resultados separados por backend: las respuestas del Mock nunca se reutilizan como reales
    "out_root": ROOT / "outputs" / "routerarena",
}


def _pool(path: Path, only: str | None) -> dict[str, dict[str, Any]]:
    models = yaml.safe_load(path.read_text())["models"]
    if only:
        wanted = [m.strip() for m in only.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in models]
        if unknown:
            raise SystemExit(f"Modelos fuera de {path}: {unknown}")
        models = {m: models[m] for m in wanted}
    return models


def _backend_factory(args, pool: dict[str, dict[str, Any]]):
    if args.backend == "mock":
        return lambda m: MockBackend(m, skill=0.5)
    if args.backend == "openai":
        if not args.endpoints:
            raise SystemExit("--backend openai requiere --endpoints (JSON {modelo: url})")
        endpoints = json.loads(Path(args.endpoints).read_text())
        return lambda m: OpenAICompatBackend(endpoints[m], m)

    def vllm(m: str):
        from generate.backends import VLLMBackend

        spec = pool[m]
        return VLLMBackend(m, spec["revision"], tensor_parallel_size=int(spec.get("tp", 1)))

    return vllm


def _evaluate(args, router: str, choices: dict[str, str], pool: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prompts = RA.load_arena_prompts(args.arena_dir, args.split)
    models = sorted(pool)
    preds = RA.build_predictions(prompts, choices, models, args.cache, include_optimality=args.optimality)
    chosen = [r for r in preds if not r["for_optimality"]]
    invalid = sum(not RA.is_valid_generation(r["generated_result"]) for r in chosen)
    if invalid:
        print(f"AVISO: {invalid}/{len(chosen)} decisiones sin generación válida (cuentan como fallo).")
    RA.write_router_files(args.arena_dir, router, models, preds)
    sizes = {m: float(pool[m].get("active_b", pool[m]["size_b"])) for m in models}
    added = RA.register_local_models(args.arena_dir, sizes, bits=args.bits)
    if added:
        print(f"Registrados en la tabla de precios de RouterArena: {added}")
    RA.refresh_prices(args.arena_dir)
    metrics = RA.evaluate(args.arena_dir, router, args.split)
    result = {
        "router": router,
        "split": args.split,
        "backend": args.backend,
        "routerarena_commit": RA.ARENA_COMMIT,
        "models": models,
        "invalid_generations": invalid,
        "metrics": metrics,
    }
    out = Path(args.out) / f"{router}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Resultados: {out}")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("setup", "generate", "candidates", "evaluate"))
    ap.add_argument("--arena-dir", type=Path, default=DEFAULTS["arena_dir"])
    ap.add_argument("--pool", type=Path, default=DEFAULTS["pool"])
    ap.add_argument("--models", help="subconjunto del pool, separado por comas")
    ap.add_argument("--cache", type=Path, help="por defecto outputs/routerarena/<backend>/cache")
    ap.add_argument("--out", type=Path, help="por defecto outputs/routerarena/<backend>")
    ap.add_argument("--split", default="sub_10", choices=tuple(RA.SPLIT_FILES))
    ap.add_argument("--backend", default="vllm", choices=("vllm", "openai", "mock"))
    ap.add_argument("--endpoints", help="JSON {modelo: url} para --backend openai")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action="store_true", help="activa el modo thinking (por defecto: off)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--router", help="nombre del router (evaluate)")
    ap.add_argument("--choices", help="JSON {global_index: modelo} (evaluate)")
    ap.add_argument("--bits", type=int, default=16, help="precisión servida, para el precio local")
    ap.add_argument("--no-optimality", dest="optimality", action="store_false")
    args = ap.parse_args(argv)
    backend_root = DEFAULTS["out_root"] / args.backend
    args.out = args.out or backend_root
    args.cache = args.cache or backend_root / "cache"

    if args.cmd == "setup":
        RA.setup(args.arena_dir)
        return

    pool = _pool(args.pool, args.models)
    if args.cmd == "generate":
        prompts = RA.load_arena_prompts(args.arena_dir, args.split)
        summary = RA.generate_candidates(
            prompts,
            sorted(pool),
            _backend_factory(args, pool),
            args.cache,
            max_tokens=args.max_tokens,
            enable_thinking=True if args.thinking else False,
            batch=args.batch,
        )
        print(json.dumps({"valid_generations": summary, "n_prompts": len(prompts)}, indent=2))
    elif args.cmd == "candidates":
        prompts = RA.load_arena_prompts(args.arena_dir, args.split)
        rows = {}
        for m in sorted(pool):
            name = "single-" + m.replace("/", "_")
            res = _evaluate(args, name, {p["global_index"]: m for p in prompts}, pool)
            rows[m] = res["metrics"]
        out = Path(args.out) / f"candidates_{args.split}.json"
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"Resumen por candidato: {out}")
    elif args.cmd == "evaluate":
        if not (args.router and args.choices):
            raise SystemExit("evaluate requiere --router y --choices")
        choices = json.loads(Path(args.choices).read_text())
        _evaluate(args, args.router, choices, pool)


if __name__ == "__main__":
    main()
