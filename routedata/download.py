"""Descarga de datasets reales desde HuggingFace.

Uso:
    python -m routedata.download --dataset all
    python -m routedata.download --dataset routerbench
    python -m routedata.download --dataset sprout
    python -m routedata.download --dataset r2bench
    python -m routedata.download --dataset llmrouterbench
    python -m routedata.download --dataset all --force  # re-descarga aunque ya existan
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Raíz del repo
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"

# Fuentes oficiales
SOURCES = {
    "routerbench": {
        "hf_id": "withmartian/routerbench",
        "url_base": "https://huggingface.co/datasets/withmartian/routerbench/resolve/main",
        "files": [
            "routerbench_0shot.pkl",
            "routerbench_5shot.pkl",
            # routerbench_raw.pkl es opcional (1.2 GB) - no lo descargamos por defecto
        ],
        "description": "RouterBench: 23 modelos, 13 tareas",
    },
    "sprout": {
        "hf_id": "CARROT-LLM-Routing/SPROUT",
        "url_base": "https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT/resolve/main",
        "files": [
            "train-00000-of-00003.parquet",
            "train-00001-of-00003.parquet",
            "train-00002-of-00003.parquet",
            "validation-00000-of-00001.parquet",
            "test-00000-of-00001.parquet",
        ],
        "description": "SPROUT: 11 modelos, con effort explícito",
    },
    "r2bench": {
        "hf_id": "JiaqiXue/R2-Bench",
        "url_base": "https://huggingface.co/datasets/JiaqiXue/R2-Bench/resolve/main",
        "files": [],  # Se llena dinámicamente al detectar la estructura
        "description": "R2-Bench: 12 modelos, robustez",
    },
    "llmrouterbench": {
        "hf_id": "NPULH/LLMRouterBench",
        "url_base": "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main",
        "files": [
            "bench-release.tar.gz",
        ],
        "description": "LLMRouterBench: 18741 prompts, 38 LLMs",
    },
}


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calcula SHA-256 de un archivo."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> bool:
    """Descarga un archivo desde una URL. Devuelve True si se descargó correctamente."""
    if dest.exists():
        log.info("Ya existe: %s (saltando)", dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    log.info("Descargando %s...", url.split("/")[-1])
    try:
        # Usamos curl para soportar progreso y retomar
        import subprocess

        result = subprocess.run(
            ["curl", "-L", "-o", str(tmp), url, "--progress-bar", "--fail"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            log.error("Fallo descarga %s: %s", url, result.stderr)
            tmp.unlink(missing_ok=True)
            return False

        # Verifica que no esté vacío
        if tmp.stat().st_size == 0:
            log.error("Archivo vacío: %s", dest.name)
            tmp.unlink(missing_ok=True)
            return False

        tmp.rename(dest)
        log.info("OK: %s (%.1f MB)", dest.name, dest.stat().st_size / (1024 * 1024))
        return True

    except subprocess.TimeoutExpired:
        log.error("Timeout descargando %s", url)
        tmp.unlink(missing_ok=True)
        return False
    except Exception as e:
        log.error("Error descargando %s: %s", url, e)
        tmp.unlink(missing_ok=True)
        return False


def download_dataset(name: str, force: bool = False) -> list[Path]:
    """Descarga un dataset completo. Devuelve lista de archivos descargados."""
    if name not in SOURCES:
        raise ValueError(f"Dataset desconocido: {name}. Opciones: {list(SOURCES.keys())}")

    src = SOURCES[name]
    dest_dir = RAW_ROOT / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = list(src["files"])

    # Para R2-Bench, detectamos los archivos dinámicamente
    if name == "r2bench":
        log.info("Detectando estructura de R2-Bench...")
        try:
            import json
            import urllib.request

            url = f"https://huggingface.co/api/datasets/{src['hf_id']}/tree/main?recursive=true"
            with urllib.request.urlopen(url) as resp:
                tree = json.load(resp)
            files = [f["path"] for f in tree if f["type"] == "file" and not f["path"].startswith(".")]
            log.info("Encontrados %d archivos en R2-Bench", len(files))
        except Exception as e:
            log.warning("No pude detectar archivos de R2-Bench: %s", e)

    if not files:
        log.warning("No hay archivos para descargar de %s", name)
        return []

    downloaded: list[Path] = []
    for f in files:
        # Si force, borra el archivo existente
        dest = dest_dir / f
        if force and dest.exists():
            dest.unlink()
            log.info("Borrado (force): %s", dest.name)

        url = f"{src['url_base']}/{f}?download=true"
        if download_file(url, dest):
            downloaded.append(dest)

    return downloaded


def download_all(force: bool = False) -> dict[str, list[Path]]:
    """Descarga todos los datasets."""
    results: dict[str, list[Path]] = {}
    for name in SOURCES:
        log.info("=== Descargando %s ===", name)
        results[name] = download_dataset(name, force=force)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga datasets de routing")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=list(SOURCES.keys()) + ["all"],
        help="Dataset a descargar (o 'all')",
    )
    parser.add_argument("--force", action="store_true", help="Re-descarga aunque ya existan")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log verboso")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.dataset == "all":
        results = download_all(force=args.force)
    else:
        results = {args.dataset: download_dataset(args.dataset, force=args.force)}

    # Imprime resumen
    print("\n=== Resumen de descargas ===")
    for name, files in results.items():
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"  {name}: {len(files)} archivos, {total_size:.1f} MB")
        for f in files:
            print(f"    {f.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
