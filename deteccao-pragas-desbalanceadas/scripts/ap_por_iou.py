#!/usr/bin/env python
"""AP médio em cada limiar de IoU (0,50 ... 0,95) na VALIDAÇÃO, para ver onde o mAP50-95 ganha ou perde.
Uso: python scripts/ap_por_iou.py plano_E01_s42 plano_E02_s42 ...   Grava runs/val_eval/ap_por_iou.json"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcc.data.yolo_io import load_yaml
from tcc.models.yolo_aspe import load_fp32
dest = ROOT / "runs" / "val_eval" / "ap_por_iou.json"
out = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
for name in sys.argv[1:]:
    info = load_yaml(ROOT / "runs" / "logs" / f"plano_{name}.yaml")
    y, ck = load_fp32(Path(info["best"]).with_name("best_fp32.pt"), device="cuda")
    m = y.val(data=info["data"], split="val", half=False, plots=False, workers=0, verbose=False)
    ap = np.asarray(m.box.all_ap)  # (classes, 10)
    out[name] = [float(v) for v in ap.mean(0)]
    print(name, " ".join(f"{v:.3f}" for v in out[name]), "| média %.4f" % ap.mean(), flush=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
