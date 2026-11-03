"""Particiones de evaluación (E1–E8): en distribución, por clusters (anti-fuga),
leave-one-domain-out, leave-one-model-out y temporal."""
from __future__ import annotations
import hashlib, re
import numpy as np

def normalized_hash(text: str) -> str:
    t = re.sub(r"[^a-z0-9áéíóúñü ]", "", text.lower()); t = re.sub(r"\s+", " ", t).strip()
    return hashlib.md5(t.encode()).hexdigest()

def dedup_groups(texts):
    """Grupo = hash del texto normalizado (casi-duplicados triviales). Para duplicados
    semánticos usa `cluster_groups` con embeddings."""
    h = [normalized_hash(t) for t in texts]
    ids = {x: i for i, x in enumerate(dict.fromkeys(h))}
    return np.array([ids[x] for x in h])

def cluster_groups(embeddings, k: int = 64, seed: int = 0):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(embeddings)

def iid_split(n: int, test_frac: float = 0.3, seed: int = 0, groups=None):
    """70/30 por defecto (como LLMRouterBench). Con `groups`, ningún grupo se reparte entre
    train y test (split por cluster)."""
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(n); cut = int(round(n * (1 - test_frac)))
        return np.sort(perm[:cut]), np.sort(perm[cut:])
    groups = np.asarray(groups); ug = rng.permutation(np.unique(groups)); test_g, cnt = [], 0
    for g in ug:
        if cnt >= n * test_frac:
            break
        test_g.append(g); cnt += int((groups == g).sum())
    te = np.isin(groups, test_g)
    return np.where(~te)[0], np.where(te)[0]

SEEDS_LLMROUTERBENCH = (42, 999, 2024, 2025, 3407)  # semillas usadas por LLMRouterBench

def leave_group_out(groups):
    """Itera (nombre, idx_train, idx_test) dejando fuera un dominio/dataset cada vez."""
    groups = np.asarray(groups)
    for g in np.unique(groups):
        yield g, np.where(groups != g)[0], np.where(groups == g)[0]

def leave_model_out(n_models: int, n_holdout: int = 1, n_splits: int = 5, seed: int = 0):
    """Subconjuntos de modelos 'no vistos' para medir generalización a modelos nuevos."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_splits):
        held = np.sort(rng.choice(n_models, size=n_holdout, replace=False))
        out.append((np.setdiff1d(np.arange(n_models), held), held))
    return out

def temporal_split(timestamps, cutoff):
    t = np.asarray(timestamps)
    return np.where(t < cutoff)[0], np.where(t >= cutoff)[0]

def prequential(snapshot_ids):
    """Para snapshots ordenados s0<s1<...: entrena con ≤si, evalúa en s(i+1)."""
    ids = list(snapshot_ids)
    for i in range(1, len(ids)):
        yield ids[:i], ids[i]
