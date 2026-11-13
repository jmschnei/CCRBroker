"""RouterArena (RouteWorks/RouterArena en HF): conjunto de SOLO EVALUACIÓN.

Hechos verificados: 8 400 preguntas (split `full`), 809 (`sub_10`) y 420 (`robustness`);
9 dominios, 44 categorías; niveles de dificultad fácil/medio/difícil. El README del
proyecto prohíbe entrenar o ajustar componentes del router con estos datos (incluidas las
etiquetas) para enviar al leaderboard. Este adaptador marca `eval_only=True` y
`assert_trainable` impide usarlo para entrenar.
Los esquemas de columnas han variado entre revisiones del dataset en HF: verifica.
"""

from __future__ import annotations

import json

import pandas as pd


def _as_list(v) -> list:
    """Options llega como lista, array de numpy, string JSON o vacío según la revisión del dataset."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return []
        try:
            parsed = json.loads(v)
            return list(parsed) if isinstance(parsed, list) else [v]
        except ValueError:
            return [v]
    try:
        return list(v)
    except TypeError:
        return [v]


def _compose_text(context, question, options: list) -> str:
    """Texto que ve el modelo: Context + Question + opciones numeradas (1., 2., ...)."""
    parts = []
    if isinstance(context, str) and context.strip():
        parts.append(context.strip())
    parts.append("" if question is None else str(question))
    if options:
        parts.append("\n".join(f"{i}. {o}" for i, o in enumerate(options, 1)))
    return "\n\n".join(parts)


def to_queries(df: pd.DataFrame) -> pd.DataFrame:
    """df con columnas 'Question','Context','Options','Answer','Domain','Category',
    'Difficulty','Global Index' (si existen). `text` incluye Context y Options; ambos se
    conservan además en columnas propias (`context`, `options` como JSON)."""
    g = lambda c, d="": df[c] if c in df.columns else d
    n = len(df)
    contexts = df["Context"].tolist() if "Context" in df.columns else [""] * n
    questions = df["Question"].tolist() if "Question" in df.columns else [""] * n
    options = [_as_list(v) for v in df["Options"]] if "Options" in df.columns else [[] for _ in range(n)]
    q = pd.DataFrame(
        {
            "query_id": ("routerarena-" + df["Global Index"].astype(str))
            if "Global Index" in df.columns
            else [f"routerarena-{i}" for i in range(len(df))],
            "text": [_compose_text(c, t, o) for c, t, o in zip(contexts, questions, options, strict=True)],
            "context": [c if isinstance(c, str) else "" for c in contexts],
            "options": [json.dumps([str(x) for x in o], ensure_ascii=False) if o else "" for o in options],
            "answer": g("Answer"),
            "domain": g("Domain", "unknown"),
            "category": g("Category", "unknown"),
            "difficulty": g("Difficulty", ""),
            "source_dataset": "routerarena",
            "language": "en",
            "eval_only": True,
        }
    )
    return q


def load_hf(split: str = "sub_10"):  # requiere `datasets` y red en TU entorno
    from datasets import load_dataset

    return to_queries(load_dataset("RouteWorks/RouterArena", split=split).to_pandas())
