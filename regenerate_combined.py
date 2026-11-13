"""Regenera los archivos unificados (por dataset + combinados + manifest).

Solo carga los 3 datasets integrados actualmente. R2-Bench queda excluido
por tamaño (ver PIPELINE_SUMMARY.md); para añadirlo, ve el adaptador
`routedata/adapters/r2bench.py`.

Reutiliza `Pipeline.save` de `routedata.pipeline` para garantizar el esquema
y el etiquetado `source_dataset` consistentes.

Uso:  python regenerate_combined.py
"""
from pathlib import Path

import pandas as pd

from routedata.adapters.llmrouterbench import LLMRouterBenchAdapter
from routedata.adapters.routerbench import load_both as load_routerbench_both
from routedata.adapters.sprout import create_sprout_adapter
from routedata.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"


def load_results() -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    return {
        "routerbench": load_routerbench_both(str(RAW / "routerbench")),
        "sprout": create_sprout_adapter(ROOT).load(),
        "llmrouterbench": LLMRouterBenchAdapter().load(str(RAW / "llmrouterbench")),
    }


def main() -> None:
    Pipeline(data_root=str(ROOT)).save(load_results())


if __name__ == "__main__":
    main()
