#!/usr/bin/env python
"""Avalia runs do plano a partir dos checkpoints fp32.

Uso: python scripts/avaliar_plano.py val  plano_E01_s42 plano_E02_s42 ...   # validação (decisões)
     python scripts/avaliar_plano.py test plano_E03_s42                     # TESTE: só ao final, uma vez
Grava runs/val_eval/plano_<split>.json com métricas globais, AP por classe, precisão e recall por classe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.data.yolo_io import load_yaml  # noqa: E402
from tcc.models.yolo_aspe import load_fp32  # noqa: E402

split, runs = sys.argv[1], sys.argv[2:]
assert split in ("val", "test")
dest = ROOT / "runs" / "val_eval" / f"plano_{split}.json"
out = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
for name in runs:
    info = load_yaml(ROOT / "runs" / "logs" / f"plano_{name}.yaml")
    y, ck = load_fp32(Path(info["best"]).with_name("best_fp32.pt"), device="cuda")
    m = y.val(data=info["data"], split=split, half=False, plots=False, workers=0, verbose=False)
    names = y.model.names
    out[name] = {
        "split": split, "epoca_do_melhor": ck["epoch"], "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "por_classe": {names[int(c)]: {"ap50": float(a50), "ap50_95": float(a), "precisao": float(p), "recall": float(r)}
                       for c, a50, a, p, r in zip(m.box.ap_class_index, m.box.ap50, m.box.ap, m.box.p, m.box.r)},
    }
    print(f"ROW {name} [{split}]: mAP50={out[name]['map50']:.4f} mAP50-95={out[name]['map50_95']:.4f} "
          f"P={out[name]['precision']:.3f} R={out[name]['recall']:.3f}", flush=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
