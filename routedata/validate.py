"""
Validación de datasets de LLM routing.

Verifica la integridad y estructura de los datos descargados en data/raw/:
  - RouterBench: data/raw/routerbench/*.pkl
  - SPROUT:      data/raw/sprout/*.parquet
  - LLMRouterBench: data/raw/llmrouterbench/extracted/**/*.json
  - R2-Bench:    data/raw/r2bench/data/*/*/*.csv

Uso:
    from routedata.validate import validate_all
    reports = validate_all()
    for name, report in reports.items():
        print(f"{name}: {'OK' if report['valid'] else 'FAIL'}")
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass
class ValidationReport:
    """Resultado de la validación de un dataset."""

    dataset: str
    valid: bool
    total_files: int = 0
    total_records: int = 0
    records_sampled: bool = False
    total_models: int = 0
    columns: list[str] = field(default_factory=list)
    expected_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sample: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "valid": self.valid,
            "total_files": self.total_files,
            "total_records": self.total_records,
            "records_sampled": self.records_sampled,
            "total_models": self.total_models,
            "columns": self.columns,
            "expected_columns": self.expected_columns,
            "missing_columns": self.missing_columns,
            "errors": self.errors,
            "warnings": self.warnings,
            "sample": self.sample,
        }


# ---------------------------------------------------------------------------
# RouterBench
# ---------------------------------------------------------------------------

ROUTERBENCH_REQUIRED_COLUMNS = {
    "sample_id",
    "prompt",
    "oracle_model_to_route_to",
}


def validate_routerbench(data_root: Path = DATA_ROOT) -> ValidationReport:
    """Valida los archivos .pkl de RouterBench."""
    report = ValidationReport(
        dataset="routerbench",
        valid=False,
        expected_columns=sorted(ROUTERBENCH_REQUIRED_COLUMNS),
    )

    raw_dir = data_root / "raw" / "routerbench"
    pkl_files = sorted(raw_dir.glob("*.pkl")) if raw_dir.exists() else []

    if not pkl_files:
        report.errors.append(f"No .pkl files found in {raw_dir}")
        return report

    report.total_files = len(pkl_files)

    try:
        import pandas as pd

        total_rows = 0
        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                df = pd.read_pickle(f)

            total_rows += len(df)
            missing = ROUTERBENCH_REQUIRED_COLUMNS - set(df.columns)
            if missing:
                report.errors.append(f"{pkl_file.name}: missing columns {sorted(missing)}")
            else:
                report.warnings.append(f"{pkl_file.name}: OK ({len(df)} rows)")

            # Collect all columns
            report.columns = sorted(set(report.columns) | set(df.columns))

            # Sample
            if report.sample is None:
                row = df.iloc[0]
                report.sample = {
                    "sample_id": str(row.get("sample_id", ""))[:50],
                    "prompt": str(row.get("prompt", ""))[:100],
                    "oracle": str(row.get("oracle_model_to_route_to", "")),
                }

        report.total_records = total_rows
        report.missing_columns = sorted(ROUTERBENCH_REQUIRED_COLUMNS - set(report.columns))
        report.valid = len(report.errors) == 0
        if report.valid:
            logger.info(f"RouterBench: OK ({total_rows} rows, {len(pkl_files)} files)")
        else:
            logger.warning(f"RouterBench: errors={report.errors[:3]}")
    except Exception as e:
        report.errors.append(f"Validation error: {e}")
        logger.exception("RouterBench validation failed")

    return report


# ---------------------------------------------------------------------------
# SPROUT
# ---------------------------------------------------------------------------

SPROUT_REQUIRED_COLUMNS = {
    "prompt",
    "golden_answer",
}


def validate_sprout(data_root: Path = DATA_ROOT) -> ValidationReport:
    """Valida los archivos .parquet de SPROUT."""
    report = ValidationReport(
        dataset="sprout",
        valid=False,
        expected_columns=sorted(SPROUT_REQUIRED_COLUMNS),
    )

    raw_dir = data_root / "raw" / "sprout"
    pq_files = sorted(raw_dir.glob("*.parquet")) if raw_dir.exists() else []

    if not pq_files:
        report.errors.append(f"No .parquet files found in {raw_dir}")
        return report

    report.total_files = len(pq_files)

    try:
        import pandas as pd

        total_rows = 0
        for pq_file in pq_files:
            df = pd.read_parquet(pq_file)
            total_rows += len(df)

            missing = SPROUT_REQUIRED_COLUMNS - set(df.columns)
            if missing:
                report.errors.append(f"{pq_file.name}: missing columns {sorted(missing)}")
            else:
                report.warnings.append(f"{pq_file.name}: OK ({len(df)} rows)")

            report.columns = sorted(set(report.columns) | set(df.columns))

            if report.sample is None:
                row = df.iloc[0]
                report.sample = {
                    "prompt": str(row.get("prompt", ""))[:100],
                    "golden_answer": str(row.get("golden_answer", ""))[:100],
                }

        report.total_records = total_rows
        report.missing_columns = sorted(SPROUT_REQUIRED_COLUMNS - set(report.columns))
        report.valid = len(report.errors) == 0
        if report.valid:
            logger.info(f"SPROUT: OK ({total_rows} rows, {len(pq_files)} files)")
        else:
            logger.warning(f"SPROUT: errors={report.errors[:3]}")
    except Exception as e:
        report.errors.append(f"Validation error: {e}")
        logger.exception("SPROUT validation failed")

    return report


# ---------------------------------------------------------------------------
# LLMRouterBench
# ---------------------------------------------------------------------------

LLMROUTERBENCH_REQUIRED_KEYS = {
    "model_name",
    "dataset_name",
    "records",
}


def validate_llmrouterbench(data_root: Path = DATA_ROOT) -> ValidationReport:
    """Valida los archivos .json de LLMRouterBench."""
    report = ValidationReport(
        dataset="llmrouterbench",
        valid=False,
        expected_columns=sorted(LLMROUTERBENCH_REQUIRED_KEYS),
    )

    extracted_dir = data_root / "raw" / "llmrouterbench" / "extracted"
    if not extracted_dir.exists():
        extracted_dir = data_root / "raw" / "llmrouterbench"

    json_files = sorted(extracted_dir.rglob("*.json")) if extracted_dir.exists() else []

    if not json_files:
        report.errors.append(f"No .json files found in {extracted_dir}")
        return report

    report.total_files = len(json_files)

    models = set()
    total_records = 0
    errors_count = 0

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            missing = LLMROUTERBENCH_REQUIRED_KEYS - set(data.keys())
            if missing:
                errors_count += 1
                report.errors.append(f"{json_file.name}: missing keys {sorted(missing)}")
                continue

            if "records" in data:
                total_records += len(data["records"])

            if "model_name" in data:
                models.add(data["model_name"])

            if report.sample is None:
                records = data.get("records", [])
                if records:
                    record = records[0]
                    report.sample = {
                        "model": data.get("model_name", ""),
                        "dataset": data.get("dataset_name", ""),
                        "instance_id": str(record.get("instance_id", ""))[:50],
                        "prompt": str(record.get("prompt", ""))[:100],
                    }

        except Exception as e:
            errors_count += 1
            report.errors.append(f"{json_file.name}: {e}")

    if errors_count:
        report.errors.append(f"{errors_count} files with errors")

    report.total_records = total_records
    report.total_models = len(models)
    report.columns = sorted(models)
    report.valid = errors_count == 0
    if report.valid:
        logger.info(f"LLMRouterBench: OK ({report.total_files} files, {len(models)} models)")
    else:
        logger.warning(f"LLMRouterBench: errors={report.errors[:3]}")

    return report


# ---------------------------------------------------------------------------
# R2-Bench
# ---------------------------------------------------------------------------

R2BENCH_REQUIRED_COLUMNS = {
    "prompts_id",
    "key",
    "original_prompt",
    "golden_answer",
    "response",
}
# correctness_score es opcional: algunos archivos solo tienen judge_raw


def validate_r2bench(data_root: Path = DATA_ROOT) -> ValidationReport:
    """Valida los archivos .csv de R2-Bench."""
    report = ValidationReport(
        dataset="r2bench",
        valid=False,
        expected_columns=sorted(R2BENCH_REQUIRED_COLUMNS),
    )

    data_dir = data_root / "raw" / "r2bench" / "data"
    if not data_dir.exists():
        report.errors.append(f"R2-Bench data directory not found: {data_dir}")
        return report

    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        report.errors.append(f"No .csv files found in {data_dir}")
        return report

    report.total_files = len(csv_files)
    report.records_sampled = True

    models = set()
    total_rows = 0
    errors_count = 0

    # Check a sample of files (first 5 per model)
    from collections import defaultdict

    model_files: dict[str, list[Path]] = defaultdict(list)
    for f in csv_files:
        # data/<org>/<model>/<budget>_judge.csv
        parts = f.parts
        # Find model name (2nd level under data/)
        if len(parts) >= 4:
            model_key = f"{parts[-3]}/{parts[-2]}"
            model_files[model_key].append(f)
        else:
            model_files["unknown"].append(f)

    checked_files = 0
    for model_key, files in model_files.items():
        for f in files[:3]:  # Check first 3 per model
            try:
                import pandas as pd

                df = pd.read_csv(f, nrows=100)  # Read first 100 rows
                missing = R2BENCH_REQUIRED_COLUMNS - set(df.columns)
                if missing:
                    errors_count += 1
                    report.errors.append(f"{f.name}: missing columns {sorted(missing)}")
                else:
                    models.add(model_key)
                    total_rows += len(df)
                    checked_files += 1

                if report.sample is None and len(df) > 0:
                    row = df.iloc[0]
                    sample = {
                        "prompts_id": row.get("prompts_id", ""),
                        "key": str(row.get("key", ""))[:30],
                        "prompt": str(row.get("original_prompt", ""))[:100],
                        "golden_answer": str(row.get("golden_answer", ""))[:100],
                    }
                    if "correctness_score" in df.columns:
                        sample["score"] = row.get("correctness_score", "")
                    report.sample = sample

            except Exception as e:
                errors_count += 1
                report.errors.append(f"{f.name}: {e}")

    if errors_count:
        report.errors.append(f"{errors_count} files with errors")

    report.total_records = total_rows
    report.total_models = len(models)
    report.columns = sorted(models)
    report.valid = errors_count == 0
    if report.valid:
        logger.info(f"R2-Bench: OK ({report.total_files} files, {len(models)} models)")
    else:
        logger.warning(f"R2-Bench: errors={report.errors[:3]}")

    return report


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def validate_all(data_root: Path = DATA_ROOT) -> dict[str, ValidationReport]:
    """Validates all datasets and returns a dict of reports."""
    reports = {}
    for name, func in [
        ("routerbench", validate_routerbench),
        ("sprout", validate_sprout),
        ("llmrouterbench", validate_llmrouterbench),
        ("r2bench", validate_r2bench),
    ]:
        reports[name] = func(data_root)

    return reports


def print_summary(reports: dict[str, ValidationReport]) -> None:
    """Prints a human-readable summary of validation results."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for name, report in reports.items():
        status = "✅ OK" if report.valid else "❌ FAIL"
        print(f"\n{name.upper():<20} {status}")
        print(f"  Files:     {report.total_files}")
        record_label = "Records sampled" if report.records_sampled else "Records"
        print(f"  {record_label:<11} {report.total_records}")
        print(f"  Models:    {report.total_models}")
        if report.columns:
            print(f"  Columns:   {', '.join(report.columns[:10])}")
        if report.sample:
            print(f"  Sample:    {report.sample}")
        if report.errors:
            print("  Errors:")
            for err in report.errors[:5]:
                print(f"    - {err}")
        if report.warnings:
            print("  Warnings:")
            for warn in report.warnings[:5]:
                print(f"    - {warn}")

    print("\n" + "=" * 70)


def validate_unified(data_root: Path = DATA_ROOT) -> dict[str, Any]:
    """Validate processed unified Parquet tables against their manifest and each other."""
    import pandas as pd

    unified_dir = data_root if data_root.name == "unified" else data_root / "processed" / "unified"
    manifest_path = unified_dir / "manifest.json"
    errors: list[str] = []
    warnings: list[str] = []
    if not manifest_path.is_file():
        return {"valid": False, "errors": [f"Manifest not found: {manifest_path}"], "warnings": []}

    try:
        manifest = json.loads(manifest_path.read_text())
        files = manifest["files"]
        queries = pd.read_parquet(unified_dir / files["queries"], columns=["query_id", "source_dataset"])
        configs = pd.read_parquet(unified_dir / files["configs"], columns=["config_id", "source_dataset"])
        responses = pd.read_parquet(
            unified_dir / files["responses"],
            columns=[
                "query_id",
                "config_id",
                "run_id",
                "source_dataset",
                "correct",
                "in_tokens",
                "out_tokens",
                "latency_s",
                "cost_usd",
                "label_source",
            ],
        )
    except (KeyError, OSError, ValueError) as error:
        return {"valid": False, "errors": [f"Cannot read unified dataset: {error}"], "warnings": []}

    counts = {"queries": len(queries), "configs": len(configs), "responses": len(responses)}
    manifest_counts = {
        "queries": manifest.get("total_queries"),
        "configs": manifest.get("total_configs"),
        "responses": manifest.get("total_responses"),
    }
    for table, actual in counts.items():
        if manifest_counts[table] != actual:
            errors.append(f"manifest {table}={manifest_counts[table]}, Parquet rows={actual}")

    for table_name, table, id_column in (
        ("queries", queries, "query_id"),
        ("configs", configs, "config_id"),
    ):
        missing_ids = int(table[id_column].isna().sum())
        duplicate_ids = int(table[id_column].duplicated().sum())
        if missing_ids:
            errors.append(f"{table_name}: {missing_ids} null {id_column} values")
        if duplicate_ids:
            errors.append(f"{table_name}: {duplicate_ids} duplicate {id_column} values")

    query_ids = set(queries["query_id"].dropna())
    config_ids = set(configs["config_id"].dropna())
    orphan_queries = int((~responses["query_id"].isin(query_ids)).sum())
    orphan_configs = int((~responses["config_id"].isin(config_ids)).sum())
    if orphan_queries:
        errors.append(f"responses: {orphan_queries} query_id foreign keys not found")
    if orphan_configs:
        errors.append(f"responses: {orphan_configs} config_id foreign keys not found")

    duplicate_responses = int(responses.duplicated(["query_id", "config_id", "run_id"]).sum())
    if duplicate_responses:
        errors.append(f"responses: {duplicate_responses} duplicate (query_id, config_id, run_id) keys")

    labeled = responses["correct"].dropna()
    out_of_range = int((~labeled.between(0, 1)).sum())
    if out_of_range:
        errors.append(f"responses: {out_of_range} correct values outside [0, 1]")
    missing_labels = int(responses["correct"].isna().sum())
    if missing_labels:
        warnings.append(f"{missing_labels} responses have no correctness label and remain NaN")

    for column in ("in_tokens", "out_tokens"):
        negatives = int(responses[column].lt(0).sum())
        if negatives:
            errors.append(f"responses: {negatives} negative values in {column}")
    for column in ("latency_s", "cost_usd"):
        if column in responses:
            negatives = int(responses[column].lt(0).sum())
            if negatives:
                errors.append(f"responses: {negatives} negative values in {column}")

    source_counts = {
        "queries": queries["source_dataset"].value_counts(dropna=False).to_dict(),
        "configs": configs["source_dataset"].value_counts(dropna=False).to_dict(),
        "responses": responses["source_dataset"].value_counts(dropna=False).to_dict(),
    }
    actual_datasets = set(queries["source_dataset"].dropna())
    manifest_datasets = set(manifest.get("datasets", []))
    if actual_datasets != manifest_datasets:
        errors.append(
            f"manifest datasets={sorted(manifest_datasets)}, query sources={sorted(actual_datasets)}"
        )

    response_query_sources = responses["query_id"].map(
        queries.drop_duplicates("query_id").set_index("query_id")["source_dataset"]
    )
    response_config_sources = responses["config_id"].map(
        configs.drop_duplicates("config_id").set_index("config_id")["source_dataset"]
    )
    query_source_mismatch = int(
        (response_query_sources.notna() & (response_query_sources != responses["source_dataset"])).sum()
    )
    config_source_mismatch = int(
        (response_config_sources.notna() & (response_config_sources != responses["source_dataset"])).sum()
    )
    if query_source_mismatch:
        errors.append(f"responses: {query_source_mismatch} source_dataset mismatches against queries")
    if config_source_mismatch:
        errors.append(f"responses: {config_source_mismatch} source_dataset mismatches against configs")

    if "latency_s" in responses and responses["latency_s"].isna().all():
        warnings.append("latency_s is absent for all unified responses")
    missing_token_counts = {
        column: int(responses[column].isna().sum()) for column in ("in_tokens", "out_tokens")
    }
    if any(missing_token_counts.values()):
        warnings.append(f"missing token counts: {missing_token_counts}")

    return {
        "valid": not errors,
        "counts": counts,
        "manifest_counts": manifest_counts,
        "source_counts": source_counts,
        "label_counts": responses["label_source"].value_counts(dropna=False).to_dict(),
        "missing_token_counts": missing_token_counts,
        "missing_labels": missing_labels,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Validate raw datasets or the processed unified tables")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Root data directory")
    parser.add_argument(
        "--unified", action="store_true", help="Validate processed Parquet tables and manifest"
    )
    args = parser.parse_args(argv)

    if args.unified:
        report = validate_unified(args.data_root)
        print("\nUNIFIED DATASET", "✅ OK" if report["valid"] else "❌ FAIL")
        for table, count in report.get("counts", {}).items():
            print(f"  {table}: {count:,} rows (manifest: {report['manifest_counts'][table]})")
        for table, counts in report.get("source_counts", {}).items():
            print(f"  {table} by source: {counts}")
        print(f"  labels: {report.get('label_counts', {})}")
        for warning in report["warnings"]:
            print(f"  WARNING: {warning}")
        for error in report["errors"]:
            print(f"  ERROR: {error}")
        if not report["valid"]:
            raise SystemExit(1)
        return

    reports = validate_all(args.data_root)
    print_summary(reports)
    if not all(report.valid for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
