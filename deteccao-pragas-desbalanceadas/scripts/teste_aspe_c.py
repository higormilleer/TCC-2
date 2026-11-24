#!/usr/bin/env python
"""Teste (C): ablação em INFERÊNCIA dos portões CA e SE do ASPE já treinado (sem retreinar), VALIDAÇÃO.
Mostra de qual parte do bloco depende o ganho em IoU alto / a perda em mAP50. Uso: python scripts/teste_aspe_c.py plano_E02_s42 plano_E04_s42"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from tcc.data.yolo_io import load_yaml
from tcc.models.yolo_aspe import load_fp32, ASPEUnit
for name in sys.argv[1:]:
    info = load_yaml(ROOT / "runs" / "logs" / f"plano_{name}.yaml")
    print(f"\n== {name}")
    for var in ("completo", "sem CA", "sem SE", "sem CA e SE"):
        y, _ = load_fp32(Path(info["best"]).with_name("best_fp32.pt"), device="cuda")
        for m in y.model.modules():
            if isinstance(m, ASPEUnit):
                if var in ("sem CA", "sem CA e SE"): m.ca.forward = lambda x: x
                if var in ("sem SE", "sem CA e SE"): m.se.forward = lambda x: x
        r = y.val(data=info["data"], split="val", half=False, plots=False, workers=0, verbose=False)
        ap = np.asarray(r.box.all_ap).mean(0)
        print(f"  {var:<12} mAP50 {r.box.map50:.4f} | mAP50-95 {r.box.map:.4f} | AP50 {ap[0]:.3f} AP70 {ap[4]:.3f} AP85 {ap[7]:.3f} AP90 {ap[8]:.3f}", flush=True)
