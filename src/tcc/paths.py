from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DOCS = ROOT / "docs"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
RUNS = ROOT / "runs"
TABLES = RUNS / "tables"
VIS = RUNS / "visualizations"
LOGS = RUNS / "logs"


def dataset_raw(name: str) -> Path:
    return RAW / name


def dataset_processed(name: str) -> Path:
    return PROCESSED / name


def split_manifest(name: str) -> Path:
    return SPLITS / name / "manifest.json"


def audit_dir(name: str) -> Path:
    return SPLITS / name / "audit"


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, SPLITS, TABLES, VIS, LOGS):
        p.mkdir(parents=True, exist_ok=True)
