"""Servidor de embeddings local compatible con la API de OpenAI (`POST /v1/embeddings`).

Sustituye a la API privada que usa LLMRouterBench upstream para `gte_Qwen2-7B-instruct`, de modo
que sus adaptadores (`generators.generator.EmbeddingGenerator`) funcionan sin modificarlos:
basta con apuntar `base_url` a este servidor.

- Modelo fijado por revisión de Hugging Face; sin prefijo de instrucción (texto crudo, como la API).
- Caché en disco (SQLite) indexada por sha256(modelo|revisión|texto): cada adaptador vuelve a
  embeber todos los prompts, así que solo la primera petición de cada texto cuesta cómputo.
- `precompute`: rellena la caché por lotes a partir de una lista de textos (mucho más rápido que
  petición a petición).

Uso:
    python scripts/embedding_server.py precompute --texts textos.jsonl
    python scripts/embedding_server.py serve --port 8765
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"
REVISION = "a8d08b36ada9cacfe34c4d6f80957772a025daf2"
SERVED_NAME = "gte_Qwen2-7B-instruct"  # nombre que piden los configs de LLMRouterBench
# Truncado en tokens del modelo. 8192 = valor usado en el Mac (MPS); en el cluster (CUDA) se puede
# subir con EMB_MAX_SEQ=32768 para no truncar más que upstream (25 000 tokens tiktoken).
DEFAULT_MAX_SEQ = int(os.environ.get("EMB_MAX_SEQ", "8192"))
# La clave de caché no incluye el truncado: cada truncado usa su propio fichero.
DEFAULT_CACHE = Path(
    os.environ.get("EMB_CACHE")
    or Path(__file__).resolve().parents[1]
    / "outputs"
    / "embeddings"
    / ("gte_qwen2_7b.sqlite" if DEFAULT_MAX_SEQ == 8192 else f"gte_qwen2_7b_L{DEFAULT_MAX_SEQ}.sqlite")
)


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, n_tokens INTEGER, vec BLOB)")
        self.lock = threading.Lock()

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(f"{MODEL}|{REVISION}|{text}".encode()).hexdigest()

    def get(self, text: str) -> tuple[list[float], int] | None:
        with self.lock:
            row = self.db.execute("SELECT n_tokens, vec FROM emb WHERE key=?", (self.key(text),)).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[1], dtype=np.float32).tolist(), int(row[0])

    def put_many(self, items: list[tuple[str, np.ndarray, int]]) -> None:
        rows = [(self.key(t), n, np.asarray(v, dtype=np.float32).tobytes()) for t, v, n in items]
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?)", rows)
            self.db.commit()

    def has(self, text: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM emb WHERE key=?", (self.key(text),)).fetchone() is not None


class Encoder:
    def __init__(self, device: str | None, max_seq_length: int):
        import torch
        from sentence_transformers import SentenceTransformer

        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.model = SentenceTransformer(
            MODEL,
            revision=REVISION,
            trust_remote_code=True,
            device=device,
            model_kwargs={"torch_dtype": torch.float16},
        )
        self.model.max_seq_length = max_seq_length
        self.tok = self.model.tokenizer
        self.lock = threading.Lock()

    def encode(self, texts: list[str], batch_size: int = 8) -> tuple[np.ndarray, list[int]]:
        n_tokens = [len(self.tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
        with self.lock:
            vecs = self.model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
            )
        return np.asarray(vecs, dtype=np.float32), n_tokens


def precompute(
    texts: list[str], cache: EmbeddingCache, enc: Encoder, batch_size: int, chunk: int = 256
) -> None:
    todo = list(dict.fromkeys(t for t in texts if t and not cache.has(t)))
    # ordenar por longitud minimiza el relleno dentro de cada lote
    todo.sort(key=len)
    print(f"{len(texts)} textos, {len(todo)} por calcular", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), chunk):
        part = todo[i : i + chunk]
        bs = batch_size
        while True:
            try:
                vecs, ntok = enc.encode(part, batch_size=bs)
                break
            except (RuntimeError, MemoryError) as e:  # OOM en CUDA / watchdog de la GPU en MPS
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(
                    f"  error con batch {bs * 2} ({type(e).__name__}); reintento con batch {bs}", flush=True
                )
                _free_accelerator_memory()
        cache.put_many(list(zip(part, vecs, ntok, strict=True)))
        done = i + len(part)
        rate = done / max(1e-9, time.time() - t0)
        print(
            f"  {done}/{len(todo)}  ({rate:.1f} textos/s, ETA {(len(todo) - done) / max(rate, 1e-9) / 60:.0f} min)",
            flush=True,
        )


def _free_accelerator_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def serve(port: int, cache: EmbeddingCache, enc_factory) -> None:
    state: dict = {"enc": None}

    def encoder() -> Encoder:
        if state["enc"] is None:
            state["enc"] = enc_factory()  # carga perezosa: si todo está en caché, no se carga el modelo
        return state["enc"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silencioso
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": SERVED_NAME, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/embeddings":
                return self._send(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            inp = req.get("input", "")
            texts = [inp] if isinstance(inp, str) else list(inp)
            out: list = [None] * len(texts)
            missing = []
            for i, t in enumerate(texts):
                hit = cache.get(t)
                if hit is None:
                    missing.append(i)
                else:
                    out[i] = hit
            if missing:
                vecs, ntok = encoder().encode([texts[i] for i in missing])
                cache.put_many([(texts[i], v, n) for i, v, n in zip(missing, vecs, ntok, strict=True)])
                for i, v, n in zip(missing, vecs, ntok, strict=True):
                    out[i] = (v.tolist(), n)
            data = [{"object": "embedding", "index": i, "embedding": v} for i, (v, _) in enumerate(out)]
            usage = sum(n for _, n in out)
            self._send(
                200,
                {
                    "object": "list",
                    "model": req.get("model", SERVED_NAME),
                    "data": data,
                    "usage": {"prompt_tokens": usage, "total_tokens": usage},
                },
            )

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Sirviendo {MODEL}@{REVISION[:7]} como '{SERVED_NAME}' en http://127.0.0.1:{port}/v1", flush=True)
    srv.serve_forever()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("serve", "precompute"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("LLMRB_EMB_PORT", "8765")))
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--device", help="mps | cuda | cpu (por defecto: mps si está disponible)")
    ap.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--texts", type=Path, help="JSONL con {'text': ...} por línea (precompute)")
    a = ap.parse_args(argv)
    cache = EmbeddingCache(a.cache)
    if a.cmd == "precompute":
        if not a.texts:
            raise SystemExit("precompute requiere --texts")
        # split("\n") y no splitlines(): algunos prompts contienen U+2028/U+2029 dentro del JSON
        texts = [json.loads(line)["text"] for line in a.texts.read_text().split("\n") if line.strip()]
        precompute(texts, cache, Encoder(a.device, a.max_seq_length), a.batch_size)
    else:
        serve(a.port, cache, lambda: Encoder(a.device, a.max_seq_length))


if __name__ == "__main__":
    main()
