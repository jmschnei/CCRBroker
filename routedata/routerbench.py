"""RouterBench (withmartian/routerbench en Hugging Face).

Hechos verificados (ficha HF): ~30 000 prompts y respuestas de 11 LLM; prompts de MBPP,
GSM8K, Winogrande, HellaSwag, MMLU, MT-Bench, etc.; hay versión 0-shot y 5-shot; los
ficheros son pickles (p. ej. routerbench_raw.pkl).
NO verificado: nombres exactos de columnas. Se asume (por el paquete oficial) que hay una
columna de prompt, una de nombre de evaluación y, por modelo, una columna de calidad y otra
'<modelo>|total_cost'. Ejecuta `python -m routedata.adapters.routerbench inspect <pkl>` primero.

SEGURIDAD: un pickle puede ejecutar código al cargarse. Verifica el hash/origen y cárgalo
en un entorno aislado.
"""

from __future__ import annotations

import sys

import pandas as pd

from .base import wide_to_tables


def load(
    path: str,
    prompt_col: str = "prompt",
    domain_col: str = "eval_name",
    id_col: str | None = "sample_id",
    source_name: str = "routerbench",
):
    df = pd.read_pickle(path)  # ¡solo ficheros de confianza!
    cost_cols = [c for c in df.columns if str(c).endswith("|total_cost")]
    models = [c[: -len("|total_cost")] for c in cost_cols]
    models = [m for m in models if m in df.columns]
    if not models:
        raise ValueError("No se detectaron columnas '<modelo>|total_cost'. Revisa con `inspect`.")
    return wide_to_tables(
        df,
        prompt_col,
        models,
        cost_suffix="|total_cost",
        domain_col=domain_col if domain_col in df.columns else None,
        id_col=id_col if id_col in df.columns else None,
        source_name=source_name,
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "inspect":
        d = pd.read_pickle(sys.argv[2])
        print(d.shape)
        print(list(d.columns))
        print(d.head(2).T)
    else:
        print(__doc__)
