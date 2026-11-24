#!/usr/bin/env python
"""Avalia os runs de reprodução do YOLO-ASPE no split de VALIDAÇÃO (holdout intocado).

Uso: python scripts/eval_yolo_aspe.py NOME_DO_RUN [NOME_DO_RUN ...]
Ex.:  python scripts/eval_yolo_aspe.py yolov8s_baseline_pt_s42 yolov8s_aspe_pt_s42
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO  # noqa: E402

from tcc.models.yolo_aspe import load_fp32  # noqa: E402  (também registra as classes)
from tcc.data.yolo_io import load_yaml  # noqa: E402

out = {}
for name in sys.argv[1:]:
    info = load_yaml(ROOT / "runs" / "logs" / f"repro_{name}.yaml")
    # Split do artigo: avalia no TEST dele (como o artigo). Split do TCC: só validação (holdout intocado).
    eval_split = "val" if info.get("split", "ours") == "ours" else "test"
    data = info.get("data", str(ROOT / "data" / "processed" / "agropest12" / "data.yaml"))
    fp32 = Path(info["best"]).with_name("best_fp32.pt")
    if not fp32.exists():
        print(f"AVISO {name}: sem best_fp32.pt; o best.pt (fp16) pode estar corrompido pelo estouro do fp16 nas variâncias do BatchNorm.")
        y, ck = YOLO(info["best"]), None
    else:
        y, ck = load_fp32(fp32, device="cuda")
    m = y.val(data=data, split=eval_split, half=False, plots=False, workers=0)
    params = sum(p.numel() for p in y.model.parameters()) / 1e6
    out[name] = {"eval_split": eval_split, "map50": float(m.box.map50), "map50_95": float(m.box.map), "params_M": round(params, 2)}
    print(f"ROW {name}: mAP50={out[name]['map50']:.4f} mAP50-95={out[name]['map50_95']:.4f} params={params:.2f}M", flush=True)
    dest = ROOT / "runs" / "val_eval" / "yolo_aspe_repro_val.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
