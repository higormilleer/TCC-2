"""Matrizes de confusão do E03 e do E05 no AgroPest-12 (validação, sementes 43 e 44 somadas).

Usa os registros por objeto da análise qualitativa (runs/val_eval/qualitativa_det.pkl): cada objeto de
referência é contado na linha da sua classe e na coluna da classe predita (acerto ou classe trocada) ou
na coluna "não detectado" (localização ruim ou objeto perdido). Só validação; o teste não é usado.
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "escrita" / "figuras"
NOMES = ["formiga", "abelha", "besouro", "lagarta", "minhoca", "tesourinha", "gafanhoto", "mariposa",
         "lesma", "caracol", "vespa", "gorgulho"]

det = pickle.load(open(ROOT / "runs" / "val_eval" / "qualitativa_det.pkl", "rb"))


def matriz(mod):
    m = np.zeros((12, 13))
    for k in ("agropest12_s43", "agropest12_s44"):
        for x in det[k][mod]:
            c, tipo, outra = x[2], x[3], x[5]
            if tipo == "ok":
                m[c, c] += 1
            elif tipo == "cls":
                m[c, outra] += 1
            else:
                m[c, 12] += 1
    return 100 * m / m.sum(axis=1, keepdims=True)


plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans"})
fig, axs = plt.subplots(1, 2, figsize=(6.9, 4.0), constrained_layout=True)
for ax, mod, titulo in zip(axs, ("o3", "o5"), ("(a) E03 (YOLO26s)", "(b) E05 (ASPE residual)")):
    m = matriz(mod)
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=100)
    for i in range(12):
        for j in range(13):
            if m[i, j] >= 0.5:
                ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=6.8,
                        color="white" if m[i, j] > 55 else "#1a1a1a")
    ax.set_xticks(range(13))
    ax.set_xticklabels(NOMES + ["não detectado"], rotation=55, ha="right")
    ax.set_yticks(range(12))
    ax.set_yticklabels(NOMES)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe de referência")
    ax.set_title(titulo, fontsize=8.5)
cb = fig.colorbar(im, ax=axs, shrink=0.7, pad=0.01)
cb.set_label("% dos objetos da classe")
fig.savefig(OUT / "qual_confusao.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "qual_confusao.pdf", bbox_inches="tight", pad_inches=0.02)
print("ok")
