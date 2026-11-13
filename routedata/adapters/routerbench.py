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
from pathlib import Path

import pandas as pd

from .base import wide_to_tables


def _from_frame(
    df: pd.DataFrame,
    prompt_col: str = "prompt",
    domain_col: str = "eval_name",
    id_col: str | None = "sample_id",
    source_name: str = "routerbench",
):
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


def load(
    path: str,
    prompt_col: str = "prompt",
    domain_col: str = "eval_name",
    id_col: str | None = "sample_id",
    source_name: str = "routerbench",
):
    """Carga UN pickle de RouterBench (0-shot o 5-shot)."""
    df = pd.read_pickle(path)  # ¡solo ficheros de confianza!
    return _from_frame(df, prompt_col, domain_col, id_col, source_name)


def load_both(raw_dir: str, shots: tuple[str, ...] = ("0shot", "5shot")):
    """Carga routerbench_{shot}.pkl para cada shot y los une en un único conjunto de tablas.

    query_id = "routerbench-{shot}/{sample_id}" para que 0-shot y 5-shot no colisionen. El 5-shot
    publicado repite algunos sample_id (14) con prompts distintos: la 2.ª aparición y siguientes
    reciben el sufijo "#dup{k}". Los configs (los mismos 11 modelos en ambos) se deduplican.
    """
    root = Path(raw_dir)
    qs, cs, rs = [], [], []
    for shot in shots:
        df = pd.read_pickle(root / f"routerbench_{shot}.pkl")  # ¡solo ficheros de confianza!
        sid = df["sample_id"].astype(str)
        k = sid.groupby(sid).cumcount()
        suffix = k.map(lambda i: f"#dup{i}" if i else "")
        df = df.assign(_query_id=f"routerbench-{shot}/" + sid + suffix)
        q, c, r = _from_frame(df, id_col="_query_id")
        qs.append(q)
        cs.append(c)
        rs.append(r)
    queries = pd.concat(qs, ignore_index=True)
    configs = pd.concat(cs, ignore_index=True).drop_duplicates("config_id", ignore_index=True)
    responses = pd.concat(rs, ignore_index=True)
    return queries, configs, responses


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "inspect":
        d = pd.read_pickle(sys.argv[2])
        print(d.shape)
        print(list(d.columns))
        print(d.head(2).T)
    else:
        print(__doc__)
