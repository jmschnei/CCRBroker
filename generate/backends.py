"""Backends de generación. Producción: vLLM con pesos abiertos (pin de revisión).
Pruebas: MockBackend determinista (sin GPU)."""
from __future__ import annotations
import hashlib, time
from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class GenRequest:
    messages: list                      # formato chat [{"role":..., "content":...}]
    max_tokens: int = 1024
    temperature: float = 0.2            # LLMRouterBench usa T=0.2, top_p=1.0
    top_p: float = 1.0
    seed: int = 0
    enable_thinking: bool | None = None # Qwen3-like; None = no aplica
    stop: list | None = None
    meta: dict = field(default_factory=dict)  # NO se envía al modelo (solo para el mock/trazas)

@dataclass
class GenResult:
    text: str
    in_tokens: int
    out_tokens: int
    reasoning_tokens: int = 0
    finish_reason: str = "stop"         # 'stop' | 'length'
    latency_s: float = 0.0

class Backend(Protocol):
    name: str
    def generate(self, reqs: list[GenRequest]) -> list[GenResult]: ...

class VLLMBackend:
    """Requiere `pip install vllm` y GPU. `revision` = commit de HF (OBLIGATORIO para
    reproducibilidad y para DriftRoute)."""
    def __init__(self, model: str, revision: str, tensor_parallel_size: int = 1,
                 dtype: str = "bfloat16", max_model_len: int = 16384, **kw):
        from vllm import LLM  # import perezoso
        self.name, self.revision = model, revision
        self.llm = LLM(model=model, revision=revision, tensor_parallel_size=tensor_parallel_size,
                       dtype=dtype, max_model_len=max_model_len, **kw)
        self.tok = self.llm.get_tokenizer()
    def _prompt(self, r: GenRequest):
        kw = {} if r.enable_thinking is None else {"enable_thinking": r.enable_thinking}
        return self.tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True, **kw)
    def generate(self, reqs):
        from vllm import SamplingParams
        t0 = time.time()
        sps = [SamplingParams(temperature=r.temperature, top_p=r.top_p, seed=r.seed,
                              max_tokens=r.max_tokens, stop=r.stop) for r in reqs]
        outs = self.llm.generate([self._prompt(r) for r in reqs], sps)
        dt = (time.time() - t0) / max(1, len(reqs))
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(GenResult(text=c.text, in_tokens=len(o.prompt_token_ids),
                                 out_tokens=len(c.token_ids), finish_reason=c.finish_reason or "stop",
                                 latency_s=dt))
        return res

class MockBackend:
    """Simulador determinista: la probabilidad de acierto depende de la habilidad del modelo,
    la dificultad de la consulta (meta['difficulty'] ∈ [0,1]), el presupuesto de tokens y
    el esfuerzo. Devuelve la respuesta correcta (meta['answer']) o una incorrecta."""
    def __init__(self, name: str, skill: float = 0.5, verbosity: float = 300.0):
        self.name, self.skill, self.verbosity = name, skill, verbosity
    def _u(self, r: GenRequest, tag: str) -> float:
        h = hashlib.md5(f"{self.name}|{r.meta.get('query_id')}|{tag}|{r.seed}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF
    def generate(self, reqs):
        out = []
        for r in reqs:
            d = float(r.meta.get("difficulty", 0.5))
            need = self.verbosity * (0.5 + d) * (3.0 if r.enable_thinking else 1.0)
            used = int(min(r.max_tokens, need * (0.8 + 0.4 * self._u(r, "len"))))
            frac = min(1.0, r.max_tokens / need)
            p = max(0.02, min(0.98, 0.5 + 3.0 * (self.skill - d))) * (0.35 + 0.65 * frac)
            if r.enable_thinking:
                p = min(0.99, p + 0.12)
            ok = self._u(r, "ok") < p
            ans = str(r.meta.get("answer", "A")) if ok else str(r.meta.get("wrong", "Z"))
            out.append(GenResult(text=f"Final answer: {ans}", in_tokens=sum(len(m["content"]) // 4 for m in r.messages),
                                 out_tokens=used, reasoning_tokens=used if r.enable_thinking else 0,
                                 finish_reason="length" if used >= r.max_tokens else "stop",
                                 latency_s=used / 80.0))
        return out
