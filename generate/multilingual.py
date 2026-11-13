"""Subconjuntos ES/DE/FR de ConfigRoute (RQ8): traducción con LLM local + controles + revisión humana.

Reglas: (1) no se traducen consultas de código; (2) las respuestas doradas de mc/math no cambian
(letra/número); (3) cada traducción pasa controles automáticos (números, nº de opciones, back-
translation) y las sospechosas van a revisión humana; (4) añade además consultas NATIVAS
(redactadas por hablantes o recogidas con licencia) para no medir solo 'inglés traducido'.
Traductor: un modelo local fuera del pool evaluado (declararlo)."""

from __future__ import annotations

import json
import re

import pandas as pd

LANG = {"es": "Spanish", "de": "German", "fr": "French"}
TPL = (
    "Translate the text to {lang}. Keep all numbers, math notation, variable names, code and "
    "option labels exactly as they are. Output only the translation.\n\n{t}"
)


def _nums(s):
    return sorted(re.findall(r"\d+(?:[.,]\d+)?", s))


def _jaccard(a, b):
    A, B = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(A & B) / max(1, len(A | B))


def check_preservation(src: str, tgt: str) -> dict:
    return dict(numbers_ok=_nums(src) == _nums(tgt), length_ratio=len(tgt) / max(1, len(src)))


def translate_queries(queries: pd.DataFrame, lang: str, gen_fn, back_fn=None, min_back_jaccard=0.45):
    """gen_fn(prompt)->texto (traductor). back_fn(prompt)->texto (retro-traducción, opcional).
    Devuelve (queries_traducidas, tabla_de_control con 'needs_review')."""
    out, ctrl = [], []
    for _, r in queries.iterrows():
        meta = json.loads(r["meta"]) if isinstance(r["meta"], str) and r["meta"] else {}
        if meta.get("task_type") == "code":
            continue
        text_t = gen_fn(TPL.format(lang=LANG[lang], t=r["text"]))
        opts = json.loads(r["options"]) if r["options"] else None
        opts_t = [gen_fn(TPL.format(lang=LANG[lang], t=o)) for o in opts] if opts else None
        c = check_preservation(r["text"], text_t)
        bj = None
        if back_fn is not None:
            bj = _jaccard(
                r["text"], back_fn(f"Translate to English. Output only the translation.\n\n{text_t}")
            )
        review = (
            (not c["numbers_ok"])
            or not (0.4 <= c["length_ratio"] <= 2.5)
            or (bj is not None and bj < min_back_jaccard)
            or (opts is not None and opts_t is not None and len(opts_t) != len(opts))
        )
        nm = dict(meta, source_query_id=r["query_id"], translated=True, translator_lang=lang)
        out.append(
            {
                **r.to_dict(),
                "query_id": f"{r['query_id']}-{lang}",
                "text": text_t,
                "language": lang,
                "options": json.dumps(opts_t) if opts_t else r["options"],
                "meta": json.dumps(nm),
            }
        )
        ctrl.append(dict(query_id=out[-1]["query_id"], **c, back_jaccard=bj, needs_review=bool(review)))
    return pd.DataFrame(out), pd.DataFrame(ctrl)


def review_sample(ctrl: pd.DataFrame, n_random: int = 100, seed: int = 0) -> pd.DataFrame:
    """Todas las marcadas + muestra aleatoria del resto, para revisión humana."""
    flagged, rest = ctrl[ctrl.needs_review], ctrl[~ctrl.needs_review]
    rnd = rest.sample(n=min(n_random, len(rest)), random_state=seed) if len(rest) else rest
    return pd.concat([flagged, rnd]).drop_duplicates("query_id")
