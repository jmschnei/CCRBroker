"""Niveles de 'esfuerzo de razonamiento' para modelos open-weights con modo thinking
(p. ej. familia Qwen3) mediante *budget forcing* en dos pasadas:
 1) generar el razonamiento con tope de `think_budget` tokens;
 2) si no terminó, cerrar el bloque de razonamiento y pedir la respuesta final.
Es una aproximación operativa a 'esfuerzo' (los modelos propietarios lo exponen nativamente)."""
from __future__ import annotations
from dataclasses import dataclass, replace
from .backends import GenRequest, GenResult

@dataclass(frozen=True)
class EffortSpec:
    name: str
    enable_thinking: bool
    think_budget: int = 0
    answer_reserve: int = 512

EFFORTS = {
    "nothink":    EffortSpec("nothink", False, 0, 1024),
    "think_low":  EffortSpec("think_low", True, 512, 512),
    "think_mid":  EffortSpec("think_mid", True, 2048, 512),
    "think_high": EffortSpec("think_high", True, 8192, 1024),
}
CLOSE_THINK = "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n</think>\n\n"

def run_with_effort(backend, reqs: list[GenRequest], spec: EffortSpec) -> list[GenResult]:
    if not spec.enable_thinking:
        rs = [replace(r, enable_thinking=False) for r in reqs]  # max_tokens lo fija el llamador (presupuesto de longitud)
        return backend.generate(rs)
    first = backend.generate([replace(r, enable_thinking=True, max_tokens=spec.think_budget,
                                      stop=["</think>"]) for r in reqs])
    out, redo_idx, redo_reqs = list(first), [], []
    for i, (r, g) in enumerate(zip(reqs, first)):
        if g.finish_reason == "length":  # razonamiento cortado → forzar cierre
            msgs = list(r.messages) + [{"role": "assistant", "content": "<think>\n" + g.text + CLOSE_THINK}]
            redo_idx.append(i)
            redo_reqs.append(replace(r, messages=msgs, enable_thinking=False, max_tokens=spec.answer_reserve))
    if redo_reqs:
        for i, g2 in zip(redo_idx, backend.generate(redo_reqs)):
            g1 = first[i]
            out[i] = GenResult(text=g2.text, in_tokens=g1.in_tokens, out_tokens=g1.out_tokens + g2.out_tokens,
                               reasoning_tokens=g1.out_tokens, finish_reason=g2.finish_reason,
                               latency_s=g1.latency_s + g2.latency_s)
    return out
