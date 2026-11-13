"""Verificadores programáticos (etiqueta 'correct' ∈ {0,1}). Para tareas abiertas usa judge.py."""
from __future__ import annotations
import re, subprocess, sys, tempfile, os

def extract_final(text: str) -> str:
    m = re.findall(r"final answer\s*[:：]\s*(.+)", text, flags=re.I)
    return m[-1].strip() if m else text.strip().splitlines()[-1].strip() if text.strip() else ""

def extract_letter(text: str):
    s = extract_final(text)
    m = re.search(r"\b([A-J])\b", s.upper())
    return m.group(1) if m else None

def _num(s: str):
    s = s.replace(",", "").replace("$", "")
    b = re.findall(r"\\boxed\{([^}]*)\}", s)
    s = b[-1] if b else s
    m = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", s)
    if not m:
        return None
    t = m[-1]
    if "/" in t:
        a, d = t.split("/"); return float(a) / float(d) if float(d) else None
    return float(t)

def verify_numeric(text, gold, rel_tol: float = 1e-6):
    p, g = _num(extract_final(text)), _num(str(gold))
    return float(p is not None and g is not None and abs(p - g) <= rel_tol * max(1.0, abs(g)))

def verify_mc(text, gold_letter):
    return float(extract_letter(text) == str(gold_letter).strip().upper())

def verify_exact(text, gold):
    n = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    return float(n(extract_final(text)) == n(str(gold)))

def run_code_tests(code: str, tests: str, timeout: float = 10.0) -> float:
    """Ejecuta código generado + tests en un subproceso. ATENCIÓN: no es un sandbox
    completo; ejecútalo SOLO en contenedor/VM aislados, sin red y sin credenciales."""
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py")
        open(f, "w").write(code + "\n\n" + tests)
        try:
            r = subprocess.run([sys.executable, f], capture_output=True, timeout=timeout, cwd=d)
            return float(r.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0.0

def verify(ttype: str, text: str, gold, tests: str | None = None):
    """Devuelve (correct, label_source) o (None, None) si requiere juez."""
    if ttype == "mc":
        return verify_mc(text, gold), "verifier"
    if ttype == "math":
        return verify_numeric(text, gold), "verifier"
    if ttype == "exact":
        return verify_exact(text, gold), "verifier"
    if ttype == "code" and tests:
        m = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
        return run_code_tests(m[-1] if m else text, tests), "verifier"
    return None, None
