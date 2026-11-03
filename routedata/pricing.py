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

def derive_selfhost_prices(gpu_usd_per_hour: float, n_gpus: int,
                           decode_tok_per_s: float, prefill_tok_per_s: float,
                           utilization: float = 0.6) -> SelfHostPrice:
    """Precio efectivo por millón de tokens.

    decode_tok_per_s / prefill_tok_per_s: *throughput agregado del servidor* (todas las
    peticiones concurrentes), medido con carga realista. utilization: fracción del tiempo
    que la GPU está ocupada (penaliza el ocio).
    """
    if min(decode_tok_per_s, prefill_tok_per_s, utilization) <= 0:
        raise ValueError("throughput y utilización deben ser > 0")
    usd_s = gpu_usd_per_hour * n_gpus / 3600.0 / utilization
    return SelfHostPrice(usd_per_mtok_in=usd_s / prefill_tok_per_s * 1e6,
                         usd_per_mtok_out=usd_s / decode_tok_per_s * 1e6)

def scale_prices(price: SelfHostPrice, factor: float) -> SelfHostPrice:
    """Escenario de sensibilidad (p. ej. 0.5, 1.5)."""
    return SelfHostPrice(price.usd_per_mtok_in * factor, price.usd_per_mtok_out * factor)

def request_cost(in_tokens: int, out_tokens: int, p: SelfHostPrice) -> float:
    """Coste USD de una petición; out_tokens incluye tokens de razonamiento."""
    return (in_tokens * p.usd_per_mtok_in + out_tokens * p.usd_per_mtok_out) / 1e6
