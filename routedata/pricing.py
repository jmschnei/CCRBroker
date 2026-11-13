"""Modelo de costes para modelos de pesos abiertos autoalojados.

Con GPU propia/alquilada no hay precio por token: se *deriva* del precio de la GPU y
del rendimiento medido. Documenta siempre (price_source, price_date) y repite los
análisis con escenarios de precio (±50 %) para comprobar sensibilidad.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelfHostPrice:
    usd_per_mtok_in: float
    usd_per_mtok_out: float


def derive_selfhost_prices(
    gpu_usd_per_hour: float,
    n_gpus: int,
    decode_tok_per_s: float,
    prefill_tok_per_s: float,
    utilization: float = 0.6,
) -> SelfHostPrice:
    """Precio efectivo por millón de tokens.

    decode_tok_per_s / prefill_tok_per_s: *throughput agregado del servidor* (todas las
    peticiones concurrentes), medido con carga realista. utilization: fracción del tiempo
    que la GPU está ocupada (penaliza el ocio).
    """
    if min(decode_tok_per_s, prefill_tok_per_s, utilization) <= 0:
        raise ValueError("throughput y utilización deben ser > 0")
    usd_s = gpu_usd_per_hour * n_gpus / 3600.0 / utilization
    return SelfHostPrice(
        usd_per_mtok_in=usd_s / prefill_tok_per_s * 1e6, usd_per_mtok_out=usd_s / decode_tok_per_s * 1e6
    )


def scale_prices(price: SelfHostPrice, factor: float) -> SelfHostPrice:
    """Escenario de sensibilidad (p. ej. 0.5, 1.5)."""
    return SelfHostPrice(price.usd_per_mtok_in * factor, price.usd_per_mtok_out * factor)


def request_cost(in_tokens: int, out_tokens: int, p: SelfHostPrice) -> float:
    """Coste USD de una petición; out_tokens incluye tokens de razonamiento."""
    return (in_tokens * p.usd_per_mtok_in + out_tokens * p.usd_per_mtok_out) / 1e6


def throughput_from_responses(responses, configs, prefill_over_decode: float = 8.0):
    """Throughput agregado por modelo medido en un run: decode_tps = suma(out_tokens)/suma(latency_s)
    (latency_s ya es tiempo de pared del lote / nº de peticiones). El prefill NO se mide por separado:
    se asume prefill_tps = prefill_over_decode × decode_tps (ajusta con una prueba dedicada)."""
    r = responses.merge(configs[["config_id", "model"]], on="config_id")
    out = {}
    for m, g in r.groupby("model"):
        lat = g["latency_s"].sum()
        if lat > 0 and g["out_tokens"].sum() > 0:
            d = float(g["out_tokens"].sum() / lat)
            out[m] = dict(decode_tps=d, prefill_tps=d * prefill_over_decode)
    return out


def reprice_configs(
    configs,
    throughput: dict,
    gpu_usd_per_hour: float,
    tp_by_model: dict | None = None,
    utilization: float = 0.6,
    price_date: str = "",
):
    """Rellena price_in/price_out de `configs` a partir de GPU-hora y throughput medido."""
    c = configs.copy()
    tp_by_model = tp_by_model or {}
    for m, t in throughput.items():
        pr = derive_selfhost_prices(
            gpu_usd_per_hour, tp_by_model.get(m, 1), t["decode_tps"], t["prefill_tps"], utilization
        )
        mask = c["model"] == m
        c.loc[mask, "price_in"], c.loc[mask, "price_out"] = pr.usd_per_mtok_in, pr.usd_per_mtok_out
        c.loc[mask, "price_source"] = (
            f"selfhost: {gpu_usd_per_hour} USD/GPU-h x {tp_by_model.get(m, 1)} GPU, medido"
        )
        if price_date:
            c.loc[mask, "price_date"] = price_date
    return c
