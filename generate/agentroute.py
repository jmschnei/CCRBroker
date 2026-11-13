"""AgentRoute (Fase 4): trayectorias multi-turno, políticas mixtas, coste con caché de prefijos y
utilidades contrafactuales por paso.

Contabilidad de coste (clave para H5). En cada paso el servidor recibe TODO el contexto. Dos modos:
  cache_mode="none"   : cada paso factura el contexto completo (el cambio de modelo no cuesta extra).
  cache_mode="prefix" : con caché automática de prefijos solo se factura lo no cacheado; la caché es
                        PROPIA DE CADA MODELO, así que al cambiar de modelo hay que re-procesar el
                        contexto en el modelo destino (salvo lo que conserve de llamadas previas si
                        retention="keep").
El coste se RECALCULA offline desde el log (tokens de contexto, de salida y modelo por paso): puedes
cambiar precios, modo de caché y retención sin volver a ejecutar episodios.

Entornos reales (τ²-Bench, ScienceWorld, SWE-bench...): implementa el protocolo `Env` en
`generate/envs/` (ver plantillas). Los contrafactuales requieren `Env.clone()`; con simuladores
externos usa `ReplayCloneMixin` (reproduce el historial de acciones) y comprueba determinismo.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np


class Env(Protocol):
    def reset(self, seed: int): ...  # -> observación (str)
    def step(
        self, action: str
    ): ...  # -> (obs:str, done:bool); errores de herramienta: obs empieza por "ERROR"
    def success(self) -> float: ...  # 0/1 (o [0,1])
    def clone(self) -> Env: ...


@dataclass
class StepLog:
    t: int
    model: str
    in_tokens: int  # tokens del PROMPT completo en este paso (contexto)
    out_tokens: int
    latency_s: float
    context_tokens: int
    switched: bool
    switch_prefill_tokens: int
    cached_tokens: int | None = None  # lo que reportó el servidor (si lo expone)
    obs_error: bool = False


# ------------------------------------------------------------------ coste
def _pp(p):
    """Precio de un modelo: tupla (in,out) o dict {'in','out','prefill_tps','decode_tps'} en USD/Mtok."""
    if isinstance(p, tuple | list):
        return dict(**{"in": p[0], "out": p[1]})
    return p


def billed_prefill(steps, cache_mode: str = "none", retention: str = "keep"):
    """Tokens de prefill facturados por paso. retention='keep': la caché de cada modelo sobrevive a las
    llamadas de otros modelos; 'evict': solo cuenta si el paso anterior fue del mismo modelo."""
    last, prev, out = {}, None, []
    for s in steps:
        L, m = s["in_tokens"], s["model"]
        if cache_mode == "none":
            b = L
        else:
            cached = last.get(m, 0) if (retention == "keep" or prev == m) else 0
            b = L - min(cached, L)
        out.append(int(b))
        last[m] = L + s["out_tokens"]
        prev = m
    return out


def candidate_prefill(history, model, ctx_next: int, cache_mode="none", retention="keep"):
    """Prefill que se facturaría si el próximo paso lo hiciera `model` con contexto `ctx_next`."""
    if cache_mode == "none":
        return int(ctx_next)
    last, prev = {}, None
    for s in history:
        last[s["model"]] = s["in_tokens"] + s["out_tokens"]
        prev = s["model"]
    cached = last.get(model, 0) if (retention == "keep" or prev == model) else 0
    return int(ctx_next - min(cached, ctx_next))


def steps_cost(steps, price, cache_mode="none", retention="keep", start: int = 0):
    """Coste USD (y latencia estimada si hay tps) de los pasos desde `start`."""
    bp = billed_prefill(steps, cache_mode, retention)
    cost = lat = 0.0
    for i in range(start, len(steps)):
        s = steps[i]
        p = _pp(price[s["model"]])
        cost += (bp[i] * p["in"] + s["out_tokens"] * p["out"]) / 1e6
        if p.get("prefill_tps") and p.get("decode_tps"):
            lat += bp[i] / p["prefill_tps"] + s["out_tokens"] / p["decode_tps"]
        else:
            lat += s["latency_s"]
    return cost, lat


@dataclass
class Episode:
    episode_id: str
    policy: str
    steps: list = field(default_factory=list)
    success: float = 0.0
    task_id: str = ""
    env_name: str = ""

    def totals(self, price: dict, cache_mode: str = "none", retention: str = "keep"):
        c, l = steps_cost(self.steps, price, cache_mode, retention)
        return dict(
            cost=c,
            latency=l,
            steps=len(self.steps),
            success=self.success,
            switches=sum(s["switched"] for s in self.steps),
        )


# ------------------------------------------------------------------ políticas: (t, historial, obs) -> modelo
def fixed_policy(model):
    return lambda t, h, o: model


def switch_every_k(models, k):
    return lambda t, h, o: models[(t // k) % len(models)]


def random_policy(models, seed=0):
    rng = np.random.default_rng(seed)
    return lambda t, h, o: models[int(rng.integers(len(models)))]


def escalation_policy(cheap: str, strong: str, k_fail: int = 2):
    """Baseline práctico: empieza barato y escala (sin volver) tras k errores de herramienta consecutivos."""
    st = {"esc": False}

    def pol(t, h, o):
        if t == 0:
            st["esc"] = False
        if not st["esc"] and len(h) >= k_fail and all(s.get("obs_error") for s in h[-k_fail:]):
            st["esc"] = True
        return strong if st["esc"] else cheap

    return pol


def approx_context(history, obs: str) -> int:
    """Estimación del contexto del próximo paso (para decidir antes de llamar al modelo)."""
    return (
        (history[-1]["in_tokens"] + history[-1]["out_tokens"] + len(obs) // 4)
        if history
        else len(obs) // 4 + 20
    )


# ------------------------------------------------------------------ estado de agente (clonable)
class AgentState:
    def __init__(self, env, backends, seed=0, system="You are a tool-using agent.", episode_id="ep"):
        self.env, self.backends, self.seed, self.episode_id = env, backends, seed, episode_id
        self.obs = env.reset(seed)
        self.msgs = [{"role": "system", "content": system}, {"role": "user", "content": self.obs}]
        self.steps, self.t, self.done, self.prev = [], 0, False, None

    def step(self, model: str, max_tokens: int = 512):
        from .backends import GenRequest

        g = self.backends[model].generate(
            [
                GenRequest(
                    messages=self.msgs,
                    max_tokens=max_tokens,
                    seed=self.seed + self.t,
                    meta={
                        "query_id": f"{self.episode_id}-{self.t}",
                        "answer": getattr(self.env, "gold_action", "ok"),
                        "difficulty": getattr(self.env, "difficulty", 0.5),
                    },
                )
            ]
        )[0]
        ctx = g.in_tokens or sum(len(m["content"]) // 4 for m in self.msgs)
        sw = self.prev is not None and model != self.prev
        self.msgs.append({"role": "assistant", "content": g.text})
        self.obs, self.done = self.env.step(g.text)
        self.msgs.append({"role": "user", "content": self.obs})
        self.steps.append(
            asdict(
                StepLog(
                    self.t,
                    model,
                    ctx,
                    g.out_tokens,
                    g.latency_s,
                    ctx,
                    sw,
                    ctx if sw else 0,
                    getattr(g, "cached_tokens", None),
                    str(self.obs).startswith("ERROR"),
                )
            )
        )
        self.prev, self.t = model, self.t + 1
        return self.steps[-1]

    def clone(self):
        n = copy.copy(self)
        n.env = self.env.clone()
        n.msgs = [dict(m) for m in self.msgs]
        n.steps = [dict(s) for s in self.steps]
        return n


def run_episode(
    env,
    backends: dict,
    policy: Callable,
    episode_id: str,
    policy_name: str,
    seed: int = 0,
    max_steps: int = 30,
    system: str = "You are a tool-using agent.",
    task_id: str = "",
    env_name: str = "",
):
    st = AgentState(env, backends, seed, system, episode_id)
    while not st.done and st.t < max_steps:
        st.step(policy(st.t, st.steps, st.obs))
    return Episode(episode_id, policy_name, st.steps, float(env.success()), task_id or episode_id, env_name)


# ------------------------------------------------------------------ contrafactuales por paso
def rollout_from(state: AgentState, first_model: str, continue_model: str, max_steps: int, seed_off: int = 0):
    """Desde un estado, elige `first_model` ahora y continúa con `continue_model` hasta el final."""
    st = state.clone()
    st.seed += seed_off
    st.episode_id += f"-r{seed_off}"
    n0 = len(st.steps)
    st.step(first_model)
    while not st.done and st.t < max_steps:
        st.step(continue_model)
    return float(st.env.success()), st.steps, n0


def epsilon_policy(ref: str, models, eps: float = 0.3, seed: int = 0):
    """Política de comportamiento para recoger estados: el modelo de referencia con prob. 1-eps, aleatorio con eps."""
    rng = np.random.default_rng(seed)
    return lambda t, h, o: models[int(rng.integers(len(models)))] if rng.random() < eps else ref


def probe_step(state: AgentState, model: str, seed_off: int = 0):
    """Muestra UN paso con `model` desde el estado y devuelve (acción válida?, tokens de salida). Etiqueta LOCAL barata:
    requiere que el entorno señale errores de herramienta/formato con obs que empieza por 'ERROR'."""
    st = state.clone()
    st.seed += seed_off
    st.episode_id += f"-p{seed_off}"
    log = st.step(model)
    return (not log["obs_error"]), log["out_tokens"]


def collect_step_utilities(
    env_factory,
    task_seed: int,
    backends,
    models,
    behavior_policy,
    reference_model: str,
    price: dict,
    n_rollouts: int = 3,
    max_steps: int = 30,
    probe_every: int = 1,
    cache_mode: str = "prefix",
    retention: str = "keep",
    task_id: str = "",
    mode: str = "local",
):
    """Para cada estado visitado por la política de comportamiento y cada modelo candidato m: usar m AHORA y continuar
    con `reference_model`. Por fila: utilidad (éxito medio de n_rollouts), tokens de salida del primer paso y coste
    esperado POSTERIOR al primer paso (`cost_after`, USD). El coste del primer paso NO se aprende: se calcula exacto en
    la decisión con el estado real de la caché (así el coste de cambiar de modelo es exacto).
    mode="local" (por defecto): utilidad = P(acción válida en este paso) con n muestras de UN solo paso (barato, sin
    continuación; cost_after=0). mode="rollout": continúa con el modelo de referencia hasta el final (caro, captura
    consecuencias a largo plazo; tenga en cuenta que el coste posterior incluye el cambio al modelo de referencia).
    COSTE rollout ≈ pasos_sondeados × n_modelos × n_rollouts × pasos_restantes (ver estimate_budget.py --agent-cf)."""
    rows = []
    st = AgentState(env_factory(), backends, task_seed, episode_id=f"cf{task_seed}")
    while not st.done and st.t < max_steps:
        beh = behavior_policy(st.t, st.steps, st.obs)
        if st.t % probe_every == 0 and mode == "local":
            for m in models:
                pr = [probe_step(st, m, seed_off=1000 * (k + 1)) for k in range(n_rollouts)]
                rows.append(
                    dict(
                        task_id=task_id or f"t{task_seed}",
                        t=st.t,
                        model=m,
                        prev_model=st.prev,
                        obs=st.obs,
                        ctx_tokens=approx_context(st.steps, st.obs),
                        utility=float(np.mean([p[0] for p in pr])),
                        out_tokens_first=float(np.mean([p[1] for p in pr])),
                        cost_after=0.0,
                    )
                )
        elif st.t % probe_every == 0:
            for m in models:
                res = [
                    rollout_from(st, m, reference_model, max_steps, seed_off=1000 * (k + 1))
                    for k in range(n_rollouts)
                ]
                rows.append(
                    dict(
                        task_id=task_id or f"t{task_seed}",
                        t=st.t,
                        model=m,
                        prev_model=st.prev,
                        obs=st.obs,
                        ctx_tokens=approx_context(st.steps, st.obs),
                        utility=float(np.mean([r[0] for r in res])),
                        out_tokens_first=float(np.mean([r[1][r[2]]["out_tokens"] for r in res])),
                        cost_after=float(
                            np.mean(
                                [steps_cost(r[1], price, cache_mode, retention, r[2] + 1)[0] for r in res]
                            )
                        ),
                    )
                )
        st.step(beh)
    return rows


# ------------------------------------------------------------------ clon por reproducción
class ReplayCloneMixin:
    """clone() reproduciendo el historial de acciones en un entorno nuevo con la misma semilla.
    Requiere un entorno DETERMINISTA dado (semilla, acciones). Si el entorno incluye un simulador de
    usuario con LLM, fija temperatura 0 y semilla, y verifica con `check_deterministic`."""

    def _make_fresh(self):
        raise NotImplementedError

    def clone(self):
        n = self._make_fresh()
        n.reset(self._seed)
        for a in self.history:
            n.step(a)
        return n


def check_deterministic(env_factory, seed: int, actions: list, n: int = 3) -> float:
    """Fracción de ejecuciones cuyas observaciones coinciden con la primera. Debe ser ≈ 1 antes de
    fiarte de los contrafactuales."""

    def run():
        e = env_factory()
        o = [e.reset(seed)]
        for a in actions:
            o.append(e.step(a)[0])
        return o

    ref = run()
    return float(np.mean([run() == ref for _ in range(n - 1)])) if n > 1 else 1.0


class ToyEnv:
    """Entorno mínimo (compatibilidad con pruebas anteriores)."""

    def __init__(self, horizon=4, difficulty=0.5):
        self.horizon, self.difficulty, self.gold_action = horizon, difficulty, "ok"

    def reset(self, seed):
        self.i, self.good, self.seed = 0, 0, seed
        return "start"

    def step(self, action):
        self.good += int("ok" in action.lower())
        self.i += 1
        return f"obs{self.i}", self.i >= self.horizon

    def success(self):
        return float(self.good >= self.horizon)

    def clone(self):
        return copy.deepcopy(self)
