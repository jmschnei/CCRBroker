"""Entorno sintético para ensayos en seco: pasos fáciles y difíciles con observaciones largas
(el contexto crece). Un paso fallido se reintenta; el éxito exige completar todos los pasos dentro del tope de pasos.
Con MockBackend, un modelo barato falla los pasos difíciles y uno fuerte los acierta: el enrutado
por paso tiene margen, y el cambio de modelo cuesta prefill cuando hay caché de prefijos."""
from __future__ import annotations
import numpy as np
from ..agentroute import ReplayCloneMixin

class SyntheticAgentEnv(ReplayCloneMixin):
    def __init__(self, horizon: int = 8, n_hard: int = 3, obs_chars: int = 2000, hard_d: float = 0.7, easy_d: float = 0.15):
        self.horizon, self.n_hard, self.obs_chars, self.hard_d, self.easy_d = horizon, n_hard, obs_chars, hard_d, easy_d
        self.gold_action = "ok"
    def _make_fresh(self): return SyntheticAgentEnv(self.horizon, self.n_hard, self.obs_chars, self.hard_d, self.easy_d)
    def _obs(self):
        hard = self.i in self.hard
        words = "analyze deeply derive rigorous proof" if hard else "lookup trivial fetch simple record"
        rng = np.random.default_rng(self._seed * 100 + self.i)
        filler = " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "table", "row", "value"], self.obs_chars // 6))
        return f"step {self.i} {words} {filler}"
    def reset(self, seed):
        self._seed = int(seed); rng = np.random.default_rng(self._seed)
        self.hard = set(int(x) for x in rng.choice(np.arange(1, self.horizon - 1), size=self.n_hard, replace=False))
        self.i, self.history = 0, []
        self.difficulty = self.hard_d if 0 in self.hard else self.easy_d
        return self._obs()
    def step(self, action):
        """Una acción válida ('ok') avanza el paso; una inválida devuelve ERROR y obliga a REINTENTAR el mismo paso
        (cuesta pasos y contexto, y el episodio tiene un tope de pasos en el ejecutor)."""
        ok = "ok" in action.lower(); self.history.append(action)
        if ok: self.i += 1
        done = self.i >= self.horizon
        self.difficulty = self.hard_d if self.i in self.hard else self.easy_d
        return ("" if ok else "ERROR: invalid action, retry. ") + (self._obs() if not done else "done"), done
    def success(self): return float(self.i >= self.horizon)
