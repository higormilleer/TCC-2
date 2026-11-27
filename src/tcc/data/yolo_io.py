from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def stem_key(path: Path) -> str:
    return path.stem


def yolo_label_path(image: Path, labels_dir: Path | None = None) -> Path:
    if labels_dir is None:
        if image.parent.name == "images":
            return image.parent.parent / "labels" / f"{image.stem}.txt"
        return image.with_suffix(".txt")
    return labels_dir / f"{image.stem}.txt"


def read_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        rows.append((cls, xc, yc, w, h))
    return rows


def write_yolo_label(path: Path, rows: Iterable[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = (xc - w / 2.0) * width
    y1 = (yc - h / 2.0) * height
    x2 = (xc + w / 2.0) * width
    y2 = (yc + h / 2.0) * height
    x1i = int(max(0, min(width - 1, round(x1))))
    y1i = int(max(0, min(height - 1, round(y1))))
    x2i = int(max(0, min(width, round(x2))))
    y2i = int(max(0, min(height, round(y2))))
    if x2i <= x1i:
        x2i = min(width, x1i + 1)
    if y2i <= y1i:
        y2i = min(height, y1i + 1)
    return x1i, y1i, x2i, y2i


def xyxy_to_xywhn(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    xc = (x1 + x2) / 2.0 / width
    yc = (y1 + y2) / 2.0 / height
    return xc, yc, w / width, h / height


def clip_xyxy(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def write_ultralytics_yaml(path: Path, root: Path, names: list[str], train: str = "images/train", val: str = "images/val", test: str = "images/test") -> None:
    dump_yaml(
        path,
        {
            "path": str(root.resolve()),
            "train": train,
            "val": val,
            "test": test,
            "nc": len(names),
            "names": names,
        },
    )
