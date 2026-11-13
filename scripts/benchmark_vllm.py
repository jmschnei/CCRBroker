"""Measure aggregate prefill/decode throughput for one pinned pool model on CUDA.

Run on a Linux host with NVIDIA CUDA and the optional vLLM dependency installed.
The script does not download data or infer GPU pricing; pass a verified hourly rate.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate.backends import GenRequest, VLLMBackend
from routedata.pricing import derive_selfhost_prices, scale_prices

DEFAULT_POOL = ROOT / "configs" / "pool_openweights.yaml"


def _build_prompt(tokenizer: Any, target_tokens: int) -> str:
    seed = "Explain the key idea, assumptions, and limitations of this topic clearly. "
    text = seed
    while len(tokenizer.encode(text)) < target_tokens:
        text += seed
    token_ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(token_ids)


def _batches(items: list[GenRequest], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact model key from configs/pool_openweights.yaml")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu-usd-hour", type=float, required=True, help="Verified cost per GPU-hour")
    parser.add_argument(
        "--gpu-price-source", required=True, help="Provider invoice/rate-card URL or reference"
    )
    parser.add_argument("--utilization", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "phase0" / "throughput.jsonl")
    args = parser.parse_args(argv)

    if min(args.requests, args.batch_size, args.input_tokens, args.max_tokens) <= 0 or args.warmup < 0:
        parser.error(
            "requests, batch-size, input-tokens y max-tokens deben ser positivos; warmup no negativo"
        )
    if args.gpu_usd_hour <= 0 or not 0 < args.utilization <= 1:
        parser.error("gpu-usd-hour debe ser > 0 y utilization debe estar en (0, 1]")
    if platform.system() != "Linux":
        raise SystemExit(
            f"El benchmark de vLLM requiere Linux con NVIDIA/CUDA; sistema detectado: {platform.system()}."
        )

    try:
        import torch
    except ImportError as error:
        raise SystemExit("Falta PyTorch; instale el stack CUDA compatible con la versión de vLLM.") from error
    if not torch.cuda.is_available():
        raise SystemExit("Fase 0 requiere CUDA/NVIDIA para vLLM; no hay dispositivo CUDA disponible.")

    pool = yaml.safe_load(args.pool.read_text())
    try:
        spec = pool["models"][args.model]
        revision = spec["revision"]
        tensor_parallel_size = int(spec.get("tp", 1))
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Modelo o configuración inválidos en {args.pool}: {args.model}") from error
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise SystemExit(f"La revisión de {args.model} no es un SHA de commit fijado: {revision}")
    if torch.cuda.device_count() < tensor_parallel_size:
        raise SystemExit(
            f"{args.model} requiere TP={tensor_parallel_size}, pero solo hay "
            f"{torch.cuda.device_count()} GPU(s) CUDA visibles."
        )

    model_load_started = time.perf_counter()
    backend = VLLMBackend(
        model=args.model,
        revision=revision,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    prompt = _build_prompt(backend.tok, args.input_tokens)
    messages = [{"role": "user", "content": prompt}]
    thinking = True if spec.get("thinking", False) else None

    def make_request(index: int) -> GenRequest:
        return GenRequest(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=args.seed + index,
            enable_thinking=thinking,
        )

    warmup_requests = [make_request(index) for index in range(args.warmup)]
    for batch in _batches(warmup_requests, args.batch_size):
        backend.generate(batch)

    measured_requests = [make_request(args.warmup + index) for index in range(args.requests)]
    input_tokens = 0
    output_tokens = 0
    started = time.perf_counter()
    for batch in _batches(measured_requests, args.batch_size):
        results = backend.generate(batch)
        input_tokens += sum(result.in_tokens for result in results)
        output_tokens += sum(result.out_tokens for result in results)
    elapsed_seconds = time.perf_counter() - started
    if elapsed_seconds <= 0 or input_tokens <= 0 or output_tokens <= 0:
        raise RuntimeError("La prueba no produjo tokens medibles.")

    gpu_names = [torch.cuda.get_device_name(index) for index in range(tensor_parallel_size)]
    prices = derive_selfhost_prices(
        gpu_usd_per_hour=args.gpu_usd_hour,
        n_gpus=tensor_parallel_size,
        decode_tok_per_s=output_tokens / elapsed_seconds,
        prefill_tok_per_s=input_tokens / elapsed_seconds,
        utilization=args.utilization,
    )
    record = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "revision": revision,
        "thinking": bool(spec.get("thinking", False)),
        "hardware": gpu_names,
        "gpu_count": tensor_parallel_size,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "vllm": importlib.metadata.version("vllm"),
        "benchmark": {
            "requests": args.requests,
            "warmup_requests": args.warmup,
            "batch_size": args.batch_size,
            "target_input_tokens": args.input_tokens,
            "max_output_tokens": args.max_tokens,
            "measured_input_tokens": input_tokens,
            "measured_output_tokens": output_tokens,
            "wall_seconds": elapsed_seconds,
            "model_load_seconds": model_load_seconds,
            "prefill_tokens_per_second": input_tokens / elapsed_seconds,
            "decode_tokens_per_second": output_tokens / elapsed_seconds,
        },
        "gpu_cost": {
            "usd_per_gpu_hour": args.gpu_usd_hour,
            "source": args.gpu_price_source,
            "utilization": args.utilization,
            "usd_per_million_input_tokens": prices.usd_per_mtok_in,
            "usd_per_million_output_tokens": prices.usd_per_mtok_out,
            "scenarios_minus_50_percent": {
                "input": scale_prices(prices, 0.5).usd_per_mtok_in,
                "output": scale_prices(prices, 0.5).usd_per_mtok_out,
            },
            "scenarios_plus_50_percent": {
                "input": scale_prices(prices, 1.5).usd_per_mtok_in,
                "output": scale_prices(prices, 1.5).usd_per_mtok_out,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\nResultado añadido a: {args.output}")


if __name__ == "__main__":
    main()
