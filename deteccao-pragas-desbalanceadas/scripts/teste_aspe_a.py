#!/usr/bin/env python
"""Testes sem treino sobre o efeito do ASPE (VALIDAÇÃO): (A) curvas por época, (B) deriva dos pesos em relação ao pré-treino."""
import csv, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))

print("=== A) métricas por época (results.csv, validação durante o treino) ===")
runs = {n: [{k.strip(): float(v) for k, v in x.items()} for x in csv.DictReader(open(ROOT / f"runs/plano/{n}/results.csv", encoding="utf-8"))]
        for n in ("plano_E01_s42", "plano_E02_s42", "plano_E03_s42", "plano_E04_s42")}
k50 = [c for c in runs["plano_E01_s42"][0] if "mAP50(B)" in c][0]; k95 = [c for c in runs["plano_E01_s42"][0] if "mAP50-95" in c][0]
for ep in (30, 50, 60, 84, 90, 100, 107):
    print(f"época {ep:>3}: " + " | ".join(f"{n[-7:-4]} {r[ep-1][k50]:.3f}/{r[ep-1][k95]:.3f}" for n, r in runs.items() if len(r) >= ep) + "   (mAP50/mAP50-95)")
for n, r in runs.items():
    b = max(range(len(r)), key=lambda i: r[i][k95]); mx50 = max(x[k50] for x in r)
    print(f"{n[-7:-4]}: melhor mAP50-95 na época {b+1} ({r[b][k95]:.4f}); melhor mAP50 de toda a corrida {mx50:.4f}; épocas rodadas {len(r)}")
# média das 20 melhores épocas (reduz efeito de 'pico' único)
for n, r in runs.items():
    t = sorted(r, key=lambda x: -x[k95])[:20]
    print(f"{n[-7:-4]}: média das 20 melhores épocas  mAP50 {np.mean([x[k50] for x in t]):.4f}  mAP50-95 {np.mean([x[k95] for x in t]):.4f}")

print("\n=== B) deriva dos pesos em relação ao pré-treino COCO (‖W−W0‖/‖W0‖), camadas do backbone ===")
from ultralytics import YOLO
sd0 = YOLO(str(ROOT / "yolo26s.pt")).model.float().state_dict()
def load(n):
    from tcc.models.yolo_aspe import load_fp32
    y, ck = load_fp32(ROOT / f"runs/plano/{n}/weights/best_fp32.pt"); return y.model.state_dict()
S = {"E03": load("plano_E03_s42"), "E04": load("plano_E04_s42")}
def drift(sd, layers):
    num = den = 0.0
    for k, v in sd.items():
        if not any(k.startswith(f"model.{i}.") for i in layers) or ".aspe." in k or k not in sd0 or sd0[k].shape != v.shape or not v.is_floating_point(): continue
        if not k.endswith(".weight") or v.ndim < 2: continue
        num += float((v - sd0[k]).norm() ** 2); den += float(sd0[k].norm() ** 2)
    return (num / den) ** .5
groups = {"stem (0,1)": [0, 1], "C3k2 com ASPE (2,4,6,8)": [2, 4, 6, 8], "convs de downsample (3,5,7)": [3, 5, 7], "SPPF+C2PSA (9,10)": [9, 10], "neck (11–22)": list(range(11, 23)), "head (23)": [23]}
for g, ls in groups.items():
    print(f"  {g:<30} E03 {drift(S['E03'], ls):.3f} | E04 {drift(S['E04'], ls):.3f}")
