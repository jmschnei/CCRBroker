"""Plantillas de prompt y restricciones de longitud ("use at most k tokens", como R2-Router)."""
from __future__ import annotations
import json

MC_TEMPLATE = ("Answer the following multiple-choice question. Finish with 'Final answer: <LETTER>'.\n\n"
               "{question}\n\n{options}")
MATH_TEMPLATE = ("Solve the problem. Finish with 'Final answer: <number or expression>'.\n\n{question}")
OPEN_TEMPLATE = "{question}"

def length_instruction(k: int | None) -> str:
    return "" if not k else f"\n\nUse at most {int(k)} tokens."

def build_messages(row, budget: int | None = None, system: str | None = None):
    """row: fila de `queries`. El tipo de tarea se deduce de `options`/`meta['task_type']`."""
    opts = row.get("options"); opts = json.loads(opts) if isinstance(opts, str) and opts else opts
    meta = row.get("meta"); meta = json.loads(meta) if isinstance(meta, str) and meta else (meta or {})
    ttype = meta.get("task_type", "mc" if opts else "open")
    if ttype == "mc":
        letters = "ABCDEFGHIJ"
        text = MC_TEMPLATE.format(question=row["text"],
                                  options="\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts)))
    elif ttype == "math":
        text = MATH_TEMPLATE.format(question=row["text"])
    else:
        text = OPEN_TEMPLATE.format(question=row["text"])
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": text + length_instruction(budget)})
    return msgs, ttype

def compliance(out_tokens, budget, slack: float = 1.1):
    """Cumplimiento de la restricción de longitud (R2-Bench: longitud ≤ 1.1×presupuesto)."""
    return out_tokens <= slack * budget
