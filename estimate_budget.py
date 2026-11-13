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

def main():
    """Entrypoint para la estimación estándar."""
    import sys
    if "--agent" in sys.argv:
        # Agent route estimation logic from the script
        ap2 = argparse.ArgumentParser(); ap2.add_argument("--agent", action="store_true")
        for k, v in dict(n_tasks=200, n_models=3, avg_steps=15, avg_ctx=6000, avg_new_per_step=600, avg_out=250, n_cf_tasks=100, cf_samples=3).items():
            ap2.add_argument(f"--{k}", type=int, default=v)
        ap2.add_argument("--decode_tps", type=float, default=2500); ap2.add_argument("--prefill_tps", type=float, default=20000)
        ap2.add_argument("--cf_mode", default="local"); ap2.add_argument("--usd_gpu_h", type=float, default=1.8)
        b = ap2.parse_args()
        r = estimate_agent_budget(b.n_tasks, b.n_models, b.avg_steps, b.avg_ctx, b.avg_new_per_step, b.avg_out, b.decode_tps, b.prefill_tps,
                                  usd_gpu_h=b.usd_gpu_h, n_cf_tasks=b.n_cf_tasks, cf_samples=b.cf_samples, cf_mode=b.cf_mode)
        for k, v in r.items(): print(f"{k}: {v['pasos']:,} pasos ≈ {v['gpu_horas']:,.1f} GPU-h ≈ {v['usd']:,.0f} USD")
    else:
        # Standard budget estimation logic from the script
        ap = argparse.ArgumentParser(); ap.add_argument("--n_queries", type=int, default=20000)
        ap.add_argument("--n_cfgs", type=int, default=60); ap.add_argument("--avg_in", type=int, default=400)
        ap.add_argument("--avg_out", type=int, default=600); ap.add_argument("--decode_tps", type=float, default=2500)
        ap.add_argument("--prefill_tps", type=float, default=20000); ap.add_argument("--n_gpus", type=int, default=1)
        ap.add_argument("--usd_gpu_h", type=float, default=1.8); a = ap.parse_args()
        r = estimate(a.n_queries, a.n_cfgs, a.avg_in, a.avg_out, a.decode_tps, a.prefill_tps, a.n_gpus, a.usd_gpu_h)
        print(f"Instancias: {r['instances']:,}  |  GPU-horas ≈ {r['gpu_hours']:,.0f}  |  USD ≈ {r['usd']:,.0f}")
        anchor = r["instances"] * ANCHOR_S_PER_INSTANCE_A800 / 3600
        print(f"Ancla empírica LLMRouterBench (15.7 s A800/instancia): {anchor:,.0f} GPU-horas (cota alta: modelos de razonamiento largo)")

if __name__ == "__main__" and "--agent" not in __import__("sys").argv:
    ap = argparse.ArgumentParser(); ap.add_argument("--n_queries", type=int, default=20000)
    ap.add_argument("--n_cfgs", type=int, default=60); ap.add_argument("--avg_in", type=int, default=400)
    ap.add_argument("--avg_out", type=int, default=600); ap.add_argument("--decode_tps", type=float, default=2500)
    ap.add_argument("--prefill_tps", type=float, default=20000); ap.add_argument("--n_gpus", type=int, default=1)
    ap.add_argument("--usd_gpu_h", type=float, default=1.8); a = ap.parse_args()
    r = estimate(a.n_queries, a.n_cfgs, a.avg_in, a.avg_out, a.decode_tps, a.prefill_tps, a.n_gpus, a.usd_gpu_h)
    print(f"Instancias: {r['instances']:,}  |  GPU-horas ≈ {r['gpu_hours']:,.0f}  |  USD ≈ {r['usd']:,.0f}")
    anchor = r["instances"] * ANCHOR_S_PER_INSTANCE_A800 / 3600
    print(f"Ancla empírica LLMRouterBench (15.7 s A800/instancia): {anchor:,.0f} GPU-horas (cota alta: modelos de razonamiento largo)")

def estimate_agent_budget(n_tasks, n_models, avg_steps, avg_ctx, avg_new_per_step, avg_out, decode_tps, prefill_tps,
                          n_gpus=1, usd_gpu_h=1.8, util=0.6, cache="prefix",
                          n_cf_tasks=0, cf_models=None, cf_samples=3, cf_mode="local", cf_probe_every=1):
    """GPU-horas de AgentRoute. Episodios fijos: n_tasks×n_models. Contrafactuales (opcional):
    local = 1 paso por (estado, modelo, muestra); rollout = pasos restantes (≈ avg_steps/2)."""
    pre = avg_new_per_step if cache == "prefix" else avg_ctx          # prefill medio facturado por paso
    sec_step = pre / prefill_tps + avg_out / decode_tps
    fixed_steps = n_tasks * n_models * avg_steps
    cf_models = cf_models or n_models
    cf_per_state = cf_models * cf_samples * (1 if cf_mode == "local" else avg_steps / 2)
    cf_steps = n_cf_tasks * (avg_steps / cf_probe_every) * cf_per_state + n_cf_tasks * avg_steps   # + comportamiento
    # un probe local desde un estado reprocesa TODO el contexto en el modelo candidato (sin caché por clon): usa avg_ctx
    cf_sec = (cf_steps * (avg_ctx / prefill_tps + avg_out / decode_tps)) if cf_mode == "local" else cf_steps * sec_step
    out = {}
    for name, steps, sec in (("episodios_fijos", fixed_steps, fixed_steps * sec_step), ("contrafactuales", cf_steps, cf_sec)):
        gh = sec / util / 3600 * n_gpus; out[name] = dict(pasos=int(steps), gpu_horas=gh, usd=gh * usd_gpu_h)
    return out

if __name__ == "__main__" and "--agent" in __import__("sys").argv:
    import sys
    ap2 = argparse.ArgumentParser(); ap2.add_argument("--agent", action="store_true")
    for k, v in dict(n_tasks=200, n_models=3, avg_steps=15, avg_ctx=6000, avg_new_per_step=600, avg_out=250, n_cf_tasks=100, cf_samples=3).items():
        ap2.add_argument(f"--{k}", type=int, default=v)
    ap2.add_argument("--decode_tps", type=float, default=2500); ap2.add_argument("--prefill_tps", type=float, default=20000)
    ap2.add_argument("--cf_mode", default="local"); ap2.add_argument("--usd_gpu_h", type=float, default=1.8)
    b = ap2.parse_args()
    r = estimate_agent_budget(b.n_tasks, b.n_models, b.avg_steps, b.avg_ctx, b.avg_new_per_step, b.avg_out, b.decode_tps, b.prefill_tps,
                              usd_gpu_h=b.usd_gpu_h, n_cf_tasks=b.n_cf_tasks, cf_samples=b.cf_samples, cf_mode=b.cf_mode)
    for k, v in r.items(): print(f"{k}: {v['pasos']:,} pasos ≈ {v['gpu_horas']:,.1f} GPU-h ≈ {v['usd']:,.0f} USD")

if __name__ == "__main__":
    main()
