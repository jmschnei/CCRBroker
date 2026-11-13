"""Fase 2: piloto, análisis de coste real, etiquetado con juez, auditoría humana y puertas de paso.

Medición clave: `latency_s` que guarda VLLMBackend es el tiempo de pared del lote dividido por el
nº de peticiones; por tanto suma(out_tokens)/suma(latency_s) = throughput AGREGADO de decodificación
del servidor (el que necesita pricing.derive_selfhost_prices y estimate_budget).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from routedata.stats import cohen_kappa

from .backends import GenRequest
from .judge import JUDGE_PROMPT, judge_vs_human, parse_score


# ------------------------------------------------------------------ análisis del piloto
def analyze_pilot(
    queries,
    configs,
    responses,
    n_full: int,
    tp_by_model: dict | None = None,
    gpu_usd_hr: float = 1.8,
    util: float = 0.6,
):
    """Por configuración: acierto, tokens, truncados, cumplimiento de longitud, tok/s agregado y
    proyección de GPU-horas/USD a `n_full` consultas. `tp_by_model`: nº de GPUs por modelo."""
    r = responses.merge(configs[["config_id", "model", "effort", "budget"]], on="config_id")
    tp_by_model = tp_by_model or {}
    rows = []
    for cid, g in r.groupby("config_id", sort=False):
        b = g["budget"].iloc[0]
        lat = g["latency_s"].sum()
        gpus = tp_by_model.get(g["model"].iloc[0], 1)
        sec_per_q = g["latency_s"].mean()
        rows.append(
            dict(
                config_id=cid,
                model=g["model"].iloc[0],
                effort=g["effort"].iloc[0],
                budget=b,
                n=len(g),
                acc=float(g["correct"].mean()),
                labeled=float(g["correct"].notna().mean()),
                out_tokens_mean=float(g["out_tokens"].mean()),
                out_tokens_p95=float(g["out_tokens"].quantile(0.95)),
                truncated=float((g["finish_reason"] == "length").mean()),
                compliance=float((g["out_tokens"] <= 1.1 * b).mean()) if pd.notna(b) else np.nan,
                decode_tps_agg=float(g["out_tokens"].sum() / lat) if lat > 0 else np.nan,
                gpu_hours_full=float(sec_per_q * n_full * gpus / util / 3600),
            )
        )
    df = pd.DataFrame(rows)
    df["usd_full"] = df["gpu_hours_full"] * gpu_usd_hr
    return df


def gates(
    report: pd.DataFrame,
    judge_metrics: dict | None = None,
    big_models=(),
    min_compliance=0.8,
    min_pearson=0.8,
    min_kappa=0.7,
    max_unlabeled=0.0,
):
    """Puertas de paso de la Fase 2 (ver PLAN). Devuelve {criterio: (ok, detalle)}."""
    out = {}
    bm = report[report.model.isin(big_models) & report.compliance.notna()]
    if len(bm):
        out["cumplimiento_longitud>=80%_modelos_grandes"] = (
            bool((bm.compliance >= min_compliance).all()),
            bm[["config_id", "compliance"]].round(2).to_dict("records"),
        )
    out["etiquetado_completo(verificador)"] = (
        bool((1 - report.labeled).max() <= max_unlabeled + 1e-9),
        "los no etiquetados pasan por juez" if (1 - report.labeled).max() > 0 else "ok",
    )
    if judge_metrics:
        out["juez_vs_humano"] = (
            judge_metrics["pearson"] >= min_pearson and judge_metrics["kappa"] >= min_kappa,
            judge_metrics,
        )
    return out


# ------------------------------------------------------------------ juez (capa de etiquetas superpuesta)
def apply_judges(queries, responses, judge_backends, batch=64, seed=0):
    """Puntúa con un conjunto de jueces las respuestas con label_source=='pending_judge'.
    Devuelve una *capa de etiquetas* (no modifica los shards): columnas query_id, config_id, run_id,
    correct (media de jueces), judge_std, label_source='judge'."""
    pend = responses[responses["label_source"] == "pending_judge"].merge(
        queries[["query_id", "text", "answer"]], on="query_id", how="left"
    )
    if pend.empty:
        return pd.DataFrame(
            columns=["query_id", "config_id", "run_id", "correct", "judge_std", "label_source"]
        )
    prompts = [
        JUDGE_PROMPT.format(question=t, reference=a, response=x)
        for t, a, x in zip(pend["text"], pend["answer"].fillna(""), pend["response_text"], strict=False)
    ]
    S = []
    for jb in judge_backends:
        sc = []
        for i in range(0, len(prompts), batch):
            res = jb.generate(
                [
                    GenRequest([{"role": "user", "content": p}], max_tokens=64, temperature=0.0, seed=seed)
                    for p in prompts[i : i + batch]
                ]
            )
            sc += [parse_score(x.text) for x in res]
        S.append([np.nan if s is None else s for s in sc])
    S = np.array(S, float)
    return pd.DataFrame(
        {
            "query_id": pend["query_id"],
            "config_id": pend["config_id"],
            "run_id": pend["run_id"],
            "correct": np.nanmean(S, axis=0),
            "judge_std": np.nanstd(S, axis=0),
            "label_source": "judge",
        }
    )


def overlay_labels(responses: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Aplica una capa de etiquetas (juez o humana) sobre `responses` sin tocar los shards."""
    if labels is None or labels.empty:
        return responses
    key = ["query_id", "config_id", "run_id"]
    m = responses.merge(labels[key + ["correct", "label_source"]], on=key, how="left", suffixes=("", "_new"))
    use = m["correct_new"].notna()
    m.loc[use, "correct"] = m.loc[use, "correct_new"]
    m.loc[use, "label_source"] = m.loc[use, "label_source_new"]
    return m.drop(columns=["correct_new", "label_source_new"])


# ------------------------------------------------------------------ auditoría humana (ciega)
def export_audit(responses, queries, labels_judge, n=500, seed=0, out_dir="audit"):
    """Hoja CSV CIEGA (sin modelo/config/etiqueta del juez) + mapa privado item_id -> (query, config)."""
    os.makedirs(out_dir, exist_ok=True)
    j = responses.merge(
        labels_judge[["query_id", "config_id", "run_id", "correct"]].rename(
            columns={"correct": "judge_score"}
        ),
        on=["query_id", "config_id", "run_id"],
    ).merge(queries[["query_id", "text", "answer", "domain", "language"]], on="query_id")
    if j.empty:
        raise ValueError("No hay respuestas juzgadas para auditar")
    per = max(1, n // max(1, j.groupby(["domain", "language"]).ngroups))
    j = pd.concat(
        [g.sample(n=min(per, len(g)), random_state=seed) for _, g in j.groupby(["domain", "language"])]
    ).head(n)
    j["item_id"] = [
        hashlib.md5(f"{q}|{c}|{r}".encode()).hexdigest()[:10]
        for q, c, r in zip(j.query_id, j.config_id, j.run_id, strict=False)
    ]
    sheet = j[["item_id", "domain", "language", "text", "answer", "response_text"]].rename(
        columns={"text": "question", "answer": "reference", "response_text": "response"}
    )
    sheet["human_score_a"] = ""
    sheet["human_score_b"] = ""  # 0..1 (o 0/1), dos anotadores
    sheet.to_csv(os.path.join(out_dir, "audit_sheet.csv"), index=False)
    j[["item_id", "query_id", "config_id", "run_id", "judge_score"]].to_csv(
        os.path.join(out_dir, "audit_key_PRIVATE.csv"), index=False
    )
    return sheet


def import_audit(sheet_path, key_path, thr=0.5):
    """Lee la hoja anotada: κ entre anotadores, y acuerdo juez–humano (media de humanos)."""
    s, k = pd.read_csv(sheet_path), pd.read_csv(key_path)
    m = s.merge(k, on="item_id").dropna(subset=["human_score_a", "human_score_b"])
    a, b = m["human_score_a"].astype(float).values, m["human_score_b"].astype(float).values
    h = (a + b) / 2
    res = {
        "kappa_annotators": float(cohen_kappa(a >= thr, b >= thr)),
        **judge_vs_human(m["judge_score"].values, h, thr),
    }
    return res, m


def human_label_layer(merged):
    """Capa de etiquetas humanas (prevalece sobre juez) para los ítems auditados."""
    return pd.DataFrame(
        {
            "query_id": merged["query_id"],
            "config_id": merged["config_id"],
            "run_id": merged["run_id"],
            "correct": ((merged["human_score_a"] + merged["human_score_b"]) / 2).astype(float),
            "label_source": "human",
        }
    )


def render_report_md(report: pd.DataFrame, gate_res: dict) -> str:
    lines = [
        "# Informe del piloto ConfigRoute\n",
        report.round(3).to_markdown(index=False)
        if hasattr(report, "to_markdown")
        else report.round(3).to_string(index=False),
        "\n## Puertas\n",
    ]
    for k, (ok, det) in gate_res.items():
        lines.append(f"- {'✅' if ok else '❌'} {k}: {det}")
    tot = report[["gpu_hours_full", "usd_full"]].sum()
    lines.append(
        f"\n**Proyección a escala completa:** ≈ {tot.gpu_hours_full:,.0f} GPU-horas, ≈ {tot.usd_full:,.0f} USD (con los supuestos indicados)."
    )
    return "\n".join(lines)
