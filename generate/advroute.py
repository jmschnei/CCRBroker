"""AdvRoute: perturbaciones benignas y ataques de rerouting (evaluación DEFENSIVA).

Modelo de amenaza (declararlo en el artículo): el atacante controla solo el texto de su
consulta (sin acceso al router en black-box; con logits/embeddings en white-box), y su objetivo
es (a) inflar el coste forzando el modelo caro o (b) degradar la calidad forzando el barato.
Referencias: Shafran et al. (arXiv:2501.01818) y RerouteGuard (arXiv:2601.21380).
Publicar un ataque exige divulgación responsable y acompañarlo de la defensa evaluada.
"""
from __future__ import annotations
import random, re, string
from typing import Callable
import numpy as np

# ------------------------------------------------------------ perturbaciones benignas
def typos(text: str, rate: float = 0.03, seed: int = 0) -> str:
    rng = random.Random(seed); cs = list(text)
    for i in range(len(cs) - 1):
        if cs[i].isalpha() and rng.random() < rate:
            cs[i], cs[i + 1] = cs[i + 1], cs[i]
    return "".join(cs)

def case_whitespace_noise(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    return re.sub(r" ", lambda m: " " * rng.choice([1, 1, 2]), text).strip() + rng.choice(["", " ", "\n"])

def pad(text: str, seed: int = 0, filler: str = "Please answer carefully. ") -> str:
    rng = random.Random(seed); k = rng.randint(1, 3)
    return (filler * k + text) if rng.random() < 0.5 else (text + " " + filler * k).strip()

def shuffle_options(options: list, answer_letter: str, seed: int = 0):
    """Baraja opciones de un MC y devuelve (nuevas_opciones, nueva_letra_correcta)."""
    letters = "ABCDEFGHIJ"; idx = list(range(len(options))); random.Random(seed).shuffle(idx)
    gold = letters.index(answer_letter.upper())
    return [options[i] for i in idx], letters[idx.index(gold)]

def paraphrase_requests(texts, backend_gen: Callable[[str], str]):
    """Paráfrasis/traducción con un LLM local: backend_gen(prompt)->texto."""
    tpl = "Paraphrase the following text keeping its exact meaning and any numbers. Output only the paraphrase.\n\n{t}"
    return [backend_gen(tpl.format(t=t)) for t in texts]

def translate_requests(texts, lang: str, backend_gen: Callable[[str], str]):
    tpl = "Translate to {lang}, preserving meaning and numbers. Output only the translation.\n\n{t}"
    return [backend_gen(tpl.format(lang=lang, t=t)) for t in texts]

PERTURBATIONS = {"typos": typos, "whitespace": case_whitespace_noise, "pad": pad}

# ------------------------------------------------------------ ataque tipo 'confounder gadget' (black-box)
def gadget_search(router_score: Callable[[list], np.ndarray], queries: list, vocab: list,
                  length: int = 10, iters: int = 200, seed: int = 0):
    """Búsqueda aleatoria/greedy de una secuencia de tokens que, antepuesta a cualquier
    consulta, MAXIMICE la puntuación de 'enviar al modelo caro' del router.

    router_score(lista_de_textos) -> array de P(strong) (black-box, solo salidas).
    Devuelve (gadget, score_medio_inicial, score_medio_final). Evalúa defensas con él;
    no lo uses contra servicios de terceros."""
    rng = random.Random(seed)
    g = [rng.choice(vocab) for _ in range(length)]
    base = float(np.mean(router_score(queries))); best = float(np.mean(router_score([" ".join(g) + " " + q for q in queries])))
    for _ in range(iters):
        j = rng.randrange(length); old = g[j]; g[j] = rng.choice(vocab)
        s = float(np.mean(router_score([" ".join(g) + " " + q for q in queries])))
        if s > best: best = s
        else: g[j] = old
    return " ".join(g), base, best

def ensure_attack_report(choice_clean, choice_attacked, C, Q):
    """Resumen de impacto: tasa de cambio de decisión, inflación de coste y caída de calidad."""
    from routedata import metrics as M
    idx = np.arange(len(choice_clean))
    return dict(flip_rate=float(np.mean(np.asarray(choice_clean) != np.asarray(choice_attacked))),
                cost_multiplier=M.cost_inflation(C, choice_clean, choice_attacked),
                quality_clean=float(Q[idx, choice_clean].mean()), quality_attacked=float(Q[idx, choice_attacked].mean()))
