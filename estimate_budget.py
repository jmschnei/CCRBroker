"""Estimación de GPU-horas y coste para generar datasets nuevos con pesos abiertos.
Todas las entradas son SUPUESTOS editables; calibra tok/s con una prueba corta en tu GPU."""
from __future__ import annotations
import argparse, yaml

ANCHOR_S_PER_INSTANCE_A800 = 1000 * 3600 / 229_600   # LLMRouterBench: ~1000 h A800 / 229 600 instancias ≈ 15.7 s (incluye razonamiento largo)

def estimate(n_queries, n_cfgs, avg_in, avg_out, decode_tps, prefill_tps, n_gpus, usd_gpu_h, util=0.6):
    inst = n_queries * n_cfgs
    sec = inst * (avg_in / prefill_tps + avg_out / decode_tps) / util   # tps = throughput AGREGADO del servidor
    gh = sec / 3600 * n_gpus
    return dict(instances=inst, gpu_hours=gh, usd=gh * usd_gpu_h)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n_queries", type=int, default=20000)
    ap.add_argument("--n_cfgs", type=int, default=60); ap.add_argument("--avg_in", type=int, default=400)
    ap.add_argument("--avg_out", type=int, default=600); ap.add_argument("--decode_tps", type=float, default=2500)
    ap.add_argument("--prefill_tps", type=float, default=20000); ap.add_argument("--n_gpus", type=int, default=1)
    ap.add_argument("--usd_gpu_h", type=float, default=1.8); a = ap.parse_args()
    r = estimate(a.n_queries, a.n_cfgs, a.avg_in, a.avg_out, a.decode_tps, a.prefill_tps, a.n_gpus, a.usd_gpu_h)
    print(f"Instancias: {r['instances']:,}  |  GPU-horas ≈ {r['gpu_hours']:,.0f}  |  USD ≈ {r['usd']:,.0f}")
    anchor = r["instances"] * ANCHOR_S_PER_INSTANCE_A800 / 3600
    print(f"Ancla empírica LLMRouterBench (15.7 s A800/instancia): {anchor:,.0f} GPU-horas (cota alta: modelos de razonamiento largo)")
