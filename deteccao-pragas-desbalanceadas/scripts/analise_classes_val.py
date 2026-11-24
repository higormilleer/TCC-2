#!/usr/bin/env python
"""Análise por classe na VALIDAÇÃO: o que faz uma classe ser fraca? (desbalanceamento, tamanho ou confusão entre classes)

Uso: python scripts/analise_classes_val.py plano_E03_s42
Grava runs/val_eval/analise_classes_<run>.json e imprime tabelas.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.data.yolo_io import load_yaml  # noqa: E402
from tcc.models.yolo_aspe import load_fp32  # noqa: E402

PLANO = ROOT / "data" / "processed" / "agropest12_plano"
NAMES = ["Ants", "Bees", "Beetles", "Caterpillars", "Earthworms", "Earwigs", "Grasshoppers", "Moths", "Slugs", "Snails", "Wasps", "Weevils"]


def ler_labels(split):
    n_inst = np.zeros(12, int); areas = [[] for _ in range(12)]; imgs = [set() for _ in range(12)]
    for f in (PLANO / "labels" / split).glob("*.txt"):
        for ln in f.read_text().splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            c, w, h = int(float(p[0])), float(p[3]), float(p[4])
            n_inst[c] += 1; areas[c].append((w * h) ** 0.5); imgs[c].add(f.stem)
    return n_inst, [np.median(a) if a else 0 for a in areas], [len(s) for s in imgs]


run = sys.argv[1]
info = load_yaml(ROOT / "runs" / "logs" / f"plano_{run}.yaml")
yolo, ck = load_fp32(Path(info["best"]).with_name("best_fp32.pt"), device="cuda")
m = yolo.val(data=info["data"], split="val", half=False, plots=False, workers=0, verbose=False)
ap50 = {int(c): float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)}
ap = {int(c): float(a) for c, a in zip(m.box.ap_class_index, m.box.ap)}
rec = {int(c): float(r) for c, r in zip(m.box.ap_class_index, m.box.r)}

# matriz de confusão com conf 0,25 (limiar usual de uso); linhas = predito, colunas = real; último índice = fundo
m2 = yolo.val(data=info["data"], split="val", half=False, plots=True, workers=0, verbose=False, conf=0.25, iou=0.7,
              project=str(ROOT / "runs" / "val_eval" / "cm"), name=run, exist_ok=True)  # plots=True é necessário para preencher a matriz
cm = m2.confusion_matrix.matrix.astype(float)

n_tr, area_tr, img_tr = ler_labels("train")
n_va, area_va, img_va = ler_labels("val")
res = {}
print(f"\n{'Classe':13s} {'inst.treino':>11s} {'inst.val':>8s} {'tam.med(val)':>12s} {'AP50':>6s} {'AP50-95':>7s} {'Recall':>6s} | {'perdidos(fundo)':>15s} {'confundido com':>30s}")
for c in range(12):
    gt = cm[:, c].sum()
    perdidos = cm[12, c] / gt if gt else 0
    col = cm[:12, c].copy(); acerto = col[c] / gt if gt else 0
    col[c] = 0
    j = int(col.argmax()); conf_com = f"{NAMES[j]} ({col[j] / gt * 100:.0f}%)" if col[j] > 0 else "-"
    res[NAMES[c]] = {"inst_treino": int(n_tr[c]), "inst_val": int(n_va[c]), "img_treino": int(img_tr[c]), "tam_mediano_val": float(area_va[c]),
                     "ap50": ap50[c], "ap50_95": ap[c], "recall": rec[c], "frac_perdidos_fundo": float(perdidos), "frac_acerto_conf025": float(acerto),
                     "maior_confusao": conf_com}
    print(f"{NAMES[c]:13s} {n_tr[c]:11d} {n_va[c]:8d} {area_va[c]:12.2f} {ap50[c]:6.3f} {ap[c]:7.3f} {rec[c]:6.3f} | {perdidos * 100:14.0f}% {conf_com:>30s}")

from scipy.stats import spearmanr  # noqa: E402

x_n = [n_tr[c] for c in range(12)]; x_a = [area_va[c] for c in range(12)]; y = [ap50[c] for c in range(12)]
r_n = spearmanr(x_n, y); r_a = spearmanr(x_a, y)
print(f"\nCorrelação de Spearman entre AP50 e nº de instâncias de treino: rho={r_n.correlation:.2f} (p={r_n.pvalue:.2f})")
print(f"Correlação de Spearman entre AP50 e tamanho mediano da caixa:  rho={r_a.correlation:.2f} (p={r_a.pvalue:.2f})")
print(f"Razão de desbalanceamento (treino): {max(n_tr) / min(n_tr):.2f}:1 (instâncias) ; {max(img_tr) / min(img_tr):.2f}:1 (imagens)")
res["_resumo"] = {"spearman_ap_vs_inst_treino": float(r_n.correlation), "p_inst": float(r_n.pvalue),
                  "spearman_ap_vs_tamanho": float(r_a.correlation), "p_tam": float(r_a.pvalue),
                  "razao_desbalanceamento_inst": float(max(n_tr) / min(n_tr))}
(ROOT / "runs" / "val_eval" / f"analise_classes_{run}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
