#!/usr/bin/env python
"""Salva as predições (conf>=0,001, após NMS, exatamente as do val) em runs/val_eval/pred/<run>/predictions.json — VALIDAÇÃO.
Uso: python scripts/dump_preds_val.py plano_E01_s42 ..."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcc.data.yolo_io import load_yaml
from tcc.models.yolo_aspe import load_fp32
for name in sys.argv[1:]:
    info = load_yaml(ROOT / "runs" / "logs" / f"plano_{name}.yaml")
    y, ck = load_fp32(Path(info["best"]).with_name("best_fp32.pt"), device="cuda")
    m = y.val(data=info["data"], split="val", half=False, plots=False, workers=0, verbose=False, save_json=True,
              project=str(ROOT / "runs" / "val_eval" / "pred"), name=name, exist_ok=True)
    print(name, "mAP50 %.4f mAP50-95 %.4f" % (m.box.map50, m.box.map), flush=True)
