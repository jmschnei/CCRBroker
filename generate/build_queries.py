"""Construcción de las consultas de ConfigRoute a partir de datasets públicos.

AVISO: los esquemas de columnas de abajo proceden de mi memoria de las fichas de Hugging Face
y NO se han verificado desde este entorno (sin red). Ejecuta `python -m generate.build_queries
inspect <nombre>` en tu máquina y ajusta `map` si difiere. Comprueba también licencia y
condiciones de uso (GPQA es de acceso restringido; no lo redistribuyas).
"""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
from routedata.splits import dedup_groups

SOURCES = {
    "mmlu_pro": dict(hf="TIGER-Lab/MMLU-Pro", split="test", task="mc", domain="knowledge",
                     map=dict(text="question", options="options", answer="answer", category="category")),
    "math500": dict(hf="HuggingFaceH4/MATH-500", split="test", task="math", domain="math",
                    map=dict(text="problem", answer="answer", category="subject", difficulty="level")),
    "gpqa_diamond": dict(hf="Idavidrein/gpqa", config="gpqa_diamond", split="train", task="gpqa", domain="science",
                         map=dict(text="Question", correct="Correct Answer",
                                  wrong=["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"])),
    "humaneval": dict(hf="openai/openai_humaneval", split="test", task="code", domain="code",
                      map=dict(text="prompt", tests="test", entry="entry_point")),
}

def _level(v):
    d = "".join(ch for ch in str(v) if ch.isdigit())
    return {"1": "easy", "2": "easy", "3": "medium", "4": "hard", "5": "hard"}.get(d[:1], "unknown")

def normalize(df: pd.DataFrame, name: str, spec: dict, seed: int = 0) -> pd.DataFrame:
    """DataFrame crudo -> tabla `queries` del esquema unificado."""
    m, task, rng = spec["map"], spec["task"], np.random.default_rng(seed)
    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        meta, options, answer, text = {"task_type": task}, None, "", str(r[m["text"]])
        if task == "mc":
            options = list(r[m["options"]]); answer = str(r[m["answer"]]).strip().upper()
        elif task == "math":
            answer = str(r[m["answer"]])
        elif task == "gpqa":
            opts = [str(r[m["correct"]])] + [str(r[c]) for c in m["wrong"]]
            perm = rng.permutation(len(opts)); options = [opts[j] for j in perm]
            answer = "ABCD"[int(np.where(perm == 0)[0][0])]; meta["task_type"] = "mc"
        elif task == "code":
            meta["tests"] = str(r[m["tests"]]) + f"\ncheck({r[m['entry']]})\n"
            text = "Complete the following Python function. Reply with the full function in a ```python code block.\n\n" + text
        rows.append(dict(
            query_id=f"{name}-{i}", text=text, options=json.dumps(options) if options else "", answer=answer,
            domain=spec["domain"], category=str(r[m["category"]]) if "category" in m else name,
            source_dataset=name, difficulty=_level(r[m["difficulty"]]) if "difficulty" in m else "unknown",
            language="en", cluster_id=-1, created_at="", eval_only=False, meta=json.dumps(meta)))
    q = pd.DataFrame(rows)
    return q.assign(cluster_id=dedup_groups(q["text"].tolist()))

def load_source(name: str, seed: int = 0) -> pd.DataFrame:   # requiere `datasets` + red en TU máquina
    from datasets import load_dataset
    spec = SOURCES[name]
    ds = load_dataset(spec["hf"], spec.get("config"), split=spec["split"])
    return normalize(ds.to_pandas(), name, spec, seed)

def drop_duplicates_across(queries: pd.DataFrame) -> pd.DataFrame:
    """Elimina casi-duplicados exactos entre fuentes (mismo texto normalizado)."""
    return queries.drop_duplicates(subset="cluster_id", keep="first").reset_index(drop=True)

def assign_empirical_difficulty(queries: pd.DataFrame, Q: np.ndarray, easy=0.8, hard=0.2) -> pd.DataFrame:
    """Dificultad empírica = tasa de acierto sobre todas las configs del pool -> easy/medium/hard.
    Úsala tras el piloto para estratificar la muestra completa de forma homogénea entre fuentes."""
    rate = np.nanmean(Q, axis=1)
    out = queries.copy(); out["difficulty"] = np.where(rate >= easy, "easy", np.where(rate <= hard, "hard", "medium"))
    out["pass_rate"] = rate; return out

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "inspect":
        from datasets import load_dataset
        sp = SOURCES[sys.argv[2]]; ds = load_dataset(sp["hf"], sp.get("config"), split=sp["split"])
        print(ds.features); print(ds[0])
    else:
        print(__doc__)
