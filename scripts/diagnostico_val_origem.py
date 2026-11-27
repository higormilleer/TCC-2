#!/usr/bin/env python
"""Diagnóstico (SÓ VALIDAÇÃO): o desempenho muda conforme a origem das imagens de validação?

No Kaggle/Roboflow, a pasta `train` oficial contém 3 versões aumentadas de cada foto (blur, ruído, rotação, brilho...),
enquanto `valid` e `test` oficiais têm as fotos originais, sem aumento. O nosso split `agropest12_plano` reagrupou
todas as 13.143 imagens (cópias juntas), então a validação mistura as duas origens. Aqui medimos cada metade.

Uso: python scripts/diagnostico_val_origem.py plano_E03_s42 [outros runs...]
Grava runs/val_eval/diagnostico_origem_val.json e imprime a tabela.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.data.yolo_io import load_yaml  # noqa: E402
from tcc.models.yolo_aspe import load_fp32  # noqa: E402

RAW = ROOT / "data" / "raw" / "agropest12"
PLANO = ROOT / "data" / "processed" / "agropest12_plano"
OUT = ROOT / "runs" / "val_eval" / "diagnostico_origem_val.json"
LISTAS = ROOT / "runs" / "val_eval" / "listas_origem"

origem = {}
for s in ("train", "valid", "test"):
    for f in os.listdir(RAW / s / "images"):
        origem[f] = s

val_files = sorted(os.listdir(PLANO / "images" / "val"))
grupos = {"originais (valid+test oficiais)": [], "aumentadas (train oficial)": []}
faltando = 0
for f in val_files:
    o = origem.get(f)
    if o is None:
        faltando += 1
        continue
    grupos["aumentadas (train oficial)" if o == "train" else "originais (valid+test oficiais)"].append(f)
print("imagens de validação:", len(val_files), "sem origem encontrada:", faltando)
for k, v in grupos.items():
    print(f"  {k}: {len(v)} imagens")

LISTAS.mkdir(parents=True, exist_ok=True)
yamls = {}
base = load_yaml(PLANO / "data.yaml")
for k, v in grupos.items():
    tag = "orig" if k.startswith("originais") else "aum"
    lst = LISTAS / f"val_{tag}.txt"
    lst.write_text("\n".join(str(PLANO / "images" / "val" / f) for f in v), encoding="utf-8")
    y = LISTAS / f"data_val_{tag}.yaml"
    y.write_text(
        f"path: {PLANO}\ntrain: images/train\nval: {lst}\ntest: {lst}\nnc: {base['nc']}\nnames: {list(base['names'])}\n",
        encoding="utf-8",
    )
    yamls[k] = y

res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
for name in sys.argv[1:]:
    info = load_yaml(ROOT / "runs" / "logs" / f"plano_{name}.yaml")
    yolo, ck = load_fp32(Path(info["best"]).with_name("best_fp32.pt"), device="cuda")
    res[name] = {}
    for k, y in yamls.items():
        m = yolo.val(data=str(y), split="val", half=False, plots=False, workers=0, verbose=False)
        names = yolo.model.names
        res[name][k] = {
            "n_imagens": len(grupos[k]), "map50": float(m.box.map50), "map50_95": float(m.box.map),
            "precision": float(m.box.mp), "recall": float(m.box.mr),
            "ap50_por_classe": {names[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)},
        }
        r = res[name][k]
        print(f"{name} | {k}: n={r['n_imagens']} mAP50={r['map50']:.4f} mAP50-95={r['map50_95']:.4f} P={r['precision']:.3f} R={r['recall']:.3f}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
