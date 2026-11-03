"""AgentRoute: trayectorias multi-turno con políticas mixtas y coste de cambio de modelo.

Por qué importa: con modelos autoalojados la caché KV es específica de cada modelo; al
cambiar de modelo a mitad de episodio hay que *re-procesar (prefill) todo el contexto*.
Este módulo registra, por paso: modelo, tokens, latencia y `switch_prefill_tokens`
(tokens de contexto que hubo que recalcular), para evaluar H5 con costes reales.

Integración con entornos reales (tau-bench/τ²-bench, ScienceWorld, SWE-bench, QA multi-salto):
implementa el protocolo `Env` (ver abajo) en un adaptador; este entorno no tiene red, así que
aquí solo se prueba con `ToyEnv`. Contrafactuales de utilidad por paso: `estimate_step_utility`
con rollouts Monte Carlo requiere `Env.clone()` (determinismo del simulador de usuario incluido).
"""
from __future__ import annotations
import copy, json, os
from dataclasses import dataclass, asdict, field
from typing import Callable, Protocol
import numpy as np

class Env(Protocol):
    def reset(self, seed: int): ...                 # -> observación (str)
    def step(self, action: str): ...                # -> (obs:str, done:bool)
    def success(self) -> float: ...                 # 0/1 (o [0,1])
    def clone(self) -> "Env": ...

@dataclass
class StepLog:
    t: int
    model: str
    in_tokens: int
    out_tokens: int
    latency_s: float
    context_tokens: int
    switched: bool
    switch_prefill_tokens: int

@dataclass
class Episode:
    episode_id: str
    policy: str
    steps: list = field(default_factory=list)
    success: float = 0.0
    def totals(self, price: dict):
        """price: {model: (usd_per_mtok_in, usd_per_mtok_out)}. Incluye el reprocesado por cambio."""
        cost = 0.0; lat = 0.0
        for s in self.steps:
            pi, po = price[s["model"]]
            cost += ((s["in_tokens"] + s["switch_prefill_tokens"]) * pi + s["out_tokens"] * po) / 1e6
            lat += s["latency_s"]
        return dict(cost=cost, latency=lat, steps=len(self.steps), success=self.success,
                    switches=sum(s["switched"] for s in self.steps))

# --- políticas: función (t, history, obs) -> nombre de modelo
def fixed_policy(model): return lambda t, h, o: model
def switch_every_k(models, k): return lambda t, h, o: models[(t // k) % len(models)]
def random_policy(models, seed=0):
    rng = np.random.default_rng(seed); return lambda t, h, o: models[int(rng.integers(len(models)))]

def run_episode(env: Env, backends: dict, policy: Callable, episode_id: str, policy_name: str,
                seed: int = 0, max_steps: int = 30, system: str = "You are a tool-using agent."):
    """backends: {modelo: Backend}. Una sola llamada de generación por paso."""
    from .backends import GenRequest
    obs = env.reset(seed); msgs = [{"role": "system", "content": system}, {"role": "user", "content": obs}]
    ep = Episode(episode_id, policy_name); prev = None
    for t in range(max_steps):
        model = policy(t, ep.steps, obs)
        ctx = sum(len(m["content"]) // 4 for m in msgs)
        g = backends[model].generate([GenRequest(messages=msgs, max_tokens=512, seed=seed + t,
                                                 meta={"query_id": f"{episode_id}-{t}", "answer": getattr(env, "gold_action", "ok"),
                                                       "difficulty": getattr(env, "difficulty", 0.5)})])[0]
        sw = prev is not None and model != prev
        ep.steps.append(asdict(StepLog(t, model, g.in_tokens, g.out_tokens, g.latency_s, ctx, sw,
                                       ctx if sw else 0)))
        msgs.append({"role": "assistant", "content": g.text})
        obs, done = env.step(g.text); msgs.append({"role": "user", "content": obs}); prev = model
        if done: break
    ep.success = float(env.success()); return ep

def estimate_step_utility(env: Env, backends, models, history_msgs, n_rollouts: int = 4,
                          continue_model: str | None = None, seed: int = 0, max_steps: int = 20):
    """Utilidad contrafactual de elegir cada modelo en el paso actual: éxito medio de
    n_rollouts que continúan con `continue_model`. COSTE ALTO (n_modelos × n_rollouts episodios
    por paso): limitar a ~2 000 trayectorias como en el plan."""
    out = {}
    for m in models:
        vals = []
        for r in range(n_rollouts):
            e = env.clone(); pol = lambda t, h, o, m=m: m if t == 0 else (continue_model or m)
            vals.append(run_episode(e, backends, pol, f"cf-{m}-{r}", f"cf-{m}", seed=seed + r, max_steps=max_steps).success)
        out[m] = float(np.mean(vals))
    return out

class ToyEnv:
    """Entorno sintético: el agente debe emitir 'ok' tras `horizon` pasos; la dificultad se
    pasa al MockBackend. Sirve solo para probar contabilidad de pasos/cambios/costes."""
    def __init__(self, horizon=4, difficulty=0.5): self.horizon, self.difficulty, self.gold_action = horizon, difficulty, "ok"
    def reset(self, seed): self.i, self.good, self.seed = 0, 0, seed; return "start"
    def step(self, action):
        self.good += int("ok" in action.lower()); self.i += 1
        return f"obs{self.i}", self.i >= self.horizon
    def success(self): return float(self.good >= self.horizon)
    def clone(self): return copy.deepcopy(self)
