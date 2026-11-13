from .synthetic import SyntheticAgentEnv


def get_env_factory(name: str, **kw):
    """Devuelve una fábrica sin argumentos que crea un entorno nuevo."""
    if name == "synthetic":
        return lambda: SyntheticAgentEnv(**kw)
    if name == "toy":
        from ..agentroute import ToyEnv

        return lambda: ToyEnv(**kw)
    from . import adapters_template as T

    cls = {
        "scienceworld": T.ScienceWorldEnv,
        "tau2": T.Tau2Env,
        "swebench": T.SWEBenchEnv,
        "hotpot": T.MultiHopQAEnv,
    }.get(name)
    if cls is None:
        raise KeyError(f"Entorno desconocido: {name}")
    raise NotImplementedError(
        f"'{name}' es solo una plantilla: implementa generate/envs/adapters_template.py::{cls.__name__}"
    )
