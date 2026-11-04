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
from routedata.adapters.routerbench import load as load_routerbench
from routedata.adapters.sprout import create_sprout_adapter
from routedata.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"


def load_results() -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    rb_0shot = load_routerbench(str(RAW / "routerbench" / "routerbench_0shot.pkl"))
    rb_5shot = load_routerbench(str(RAW / "routerbench" / "routerbench_5shot.pkl"))
    return {
        "routerbench": (
            pd.concat([rb_0shot[0], rb_5shot[0]], ignore_index=True),
            pd.concat([rb_0shot[1], rb_5shot[1]], ignore_index=True),
            pd.concat([rb_0shot[2], rb_5shot[2]], ignore_index=True),
        ),
        "sprout": create_sprout_adapter(ROOT).load(),
        "llmrouterbench": LLMRouterBenchAdapter().load(str(RAW / "llmrouterbench")),
    }


def main() -> None:
    Pipeline(data_root=str(ROOT)).save(load_results())


if __name__ == "__main__":
    main()
