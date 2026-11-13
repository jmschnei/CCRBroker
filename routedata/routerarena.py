"""RouterArena (RouteWorks/RouterArena en HF): conjunto de SOLO EVALUACIÓN.

Hechos verificados: 8 400 preguntas (split `full`), 809 (`sub_10`) y 420 (`robustness`);
9 dominios, 44 categorías; niveles de dificultad fácil/medio/difícil. El README del
proyecto prohíbe entrenar o ajustar componentes del router con estos datos (incluidas las
etiquetas) para enviar al leaderboard. Este adaptador marca `eval_only=True` y
`assert_trainable` impide usarlo para entrenar.
Los esquemas de columnas han variado entre revisiones del dataset en HF: verifica.
"""
from __future__ import annotations
import pandas as pd

def to_queries(df: pd.DataFrame) -> pd.DataFrame:
    """df con columnas 'Question','Context','Options','Answer','Domain','Category',
    'Difficulty','Global Index' (si existen)."""
    g = lambda c, d="": df[c] if c in df.columns else d
    q = pd.DataFrame({
        "query_id": ("routerarena-" + df["Global Index"].astype(str)) if "Global Index" in df.columns
                    else [f"routerarena-{i}" for i in range(len(df))],
        "text": df["Question"].astype(str) if "Question" in df.columns else "",
        "answer": g("Answer"), "domain": g("Domain", "unknown"), "category": g("Category", "unknown"),
        "difficulty": g("Difficulty", ""), "source_dataset": "routerarena",
        "language": "en", "eval_only": True,
    })
    return q

def load_hf(split: str = "sub_10"):  # requiere `datasets` y red en TU entorno
    from datasets import load_dataset
    return to_queries(load_dataset("RouteWorks/RouterArena", split=split).to_pandas())
