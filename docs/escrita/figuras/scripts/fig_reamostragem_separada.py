"""Versão separada da figura de reamostragem (Etapa 2C da reestruturação):
(a) reamostragem_conceito: esquema de sub e sobreamostragem (vai para a Fundamentação, Cap. 2);
(b) rfs_ip102: RFS calculado no treino do IP102 usado no TCC (vai para Material e Métodos, Cap. 4)."""
import os, json
import numpy as np
from matplotlib.patches import Rectangle
from estilo import *

LAB = os.path.join(RAIZ, "data", "processed", "ip102_limpo", "labels", "train")
cache = "ip102_train_sets.json"
if os.path.exists(cache):
    sets = json.load(open(cache))
else:
    sets = []
    for fn in os.listdir(LAB):
        if fn.endswith(".txt"):
            sets.append(sorted({int(l.split()[0]) for l in open(os.path.join(LAB, fn)) if l.strip()}))
    json.dump(sets, open(cache, "w"))
N = len(sets)
imgs = {}
for s in sets:
    for c in s:
        imgs[c] = imgs.get(c, 0) + 1
t = 54 / 11924                      # limiar do TCC: mediana das frequências (configs/minoritarias_ip102_limpo.yaml)
r_c = {c: max(1.0, np.sqrt(t / (n / N))) for c, n in imgs.items()}
r_i = [max(r_c[c] for c in s) if s else 1.0 for s in sets]
esc = N / sum(r_i)                  # mesmo número de imagens por época
apres = {c: 0.0 for c in imgs}
for s, r in zip(sets, r_i):
    for c in s:
        apres[c] += r * esc
ordem = sorted(imgs, key=lambda c: -imgs[c])
n0 = np.array([imgs[c] for c in ordem]); n1 = np.array([apres[c] for c in ordem])
print("N", N, "classes", len(ordem), "max r_c", round(max(r_c.values()), 2), "escala", round(esc, 3))
print("mais rara: antes", n0[-1], "depois", round(n1[-1], 1), "| mais frequente: antes", n0[0], "depois", round(n1[0], 1))
print("classes com r_c > 1:", sum(v > 1 for v in r_c.values()))

# ---------------- (a) esquema conceitual ----------------
fig = plt.figure(figsize=(3.6, 2.6))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, 40); ax.set_ylim(-4, 29); ax.axis("off")
MAJ, MIN = AZUL[1], "#D9822B"
def coluna(x, n, cor, extra=0, removidos=0):
    for k in range(n):
        ax.add_patch(Rectangle((x, k * 1.6), 3.2, 1.35, fc=cor, ec="white", lw=0.5))
    for k in range(n, n + extra):
        ax.add_patch(Rectangle((x, k * 1.6), 3.2, 1.35, fc="white", ec=cor, lw=0.8, hatch="////"))
    for k in range(n, n + removidos):
        ax.add_patch(Rectangle((x, k * 1.6), 3.2, 1.35, fc="none", ec=cor, lw=0.7, ls=(0, (2, 1.5))))
grupos = [("Original", 1.0, 12, 0, 3, 0), ("Subamostragem", 15.5, 3, 9, 3, 0), ("Sobreamostragem", 30.0, 12, 0, 3, 9)]
for nome, x, nmaj, rem, nmin, extra in grupos:
    coluna(x, nmaj, MAJ, removidos=rem)
    coluna(x + 4.3, nmin, MIN, extra=extra)
    ax.text(x + 3.75, -1.2, nome, ha="center", va="top", fontsize=6.8, weight="bold")
ax.add_patch(Rectangle((1.0, 25.4), 1.6, 1.2, fc=MAJ, ec="none")); ax.text(3.1, 26.0, "classe majoritária", va="center", fontsize=6.5)
ax.add_patch(Rectangle((21.0, 25.4), 1.6, 1.2, fc=MIN, ec="none")); ax.text(23.1, 26.0, "classe minoritária", va="center", fontsize=6.5)
ax.add_patch(Rectangle((1.0, 22.6), 1.6, 1.2, fc="none", ec=CINZA, ls=(0, (2, 1.5)), lw=0.7)); ax.text(3.1, 23.2, "amostra descartada", va="center", fontsize=6.5)
ax.add_patch(Rectangle((21.0, 22.6), 1.6, 1.2, fc="white", ec=CINZA, hatch="////", lw=0.7)); ax.text(23.1, 23.2, "amostra repetida", va="center", fontsize=6.5)
salvar(fig, "reamostragem_conceito")

# ---------------- (b) RFS no IP102 ----------------
fig = plt.figure(figsize=(5.2, 2.5))
bx = fig.add_axes([0.11, 0.17, 0.86, 0.78])
x = np.arange(1, len(ordem) + 1)
bx.bar(x, n0, width=0.85, color="#C3C9D1", label="imagens no treino", zorder=2)
bx.plot(x, n1, color=VERMELHO, lw=1.3, label="imagens apresentadas por época com RFS", zorder=3)
bx.axhline(54, color=TINTA, lw=0.7, ls=(0, (4, 2)), zorder=4)
bx.text(len(x) + 0.5, 54 * 1.12, "limiar t (54 imagens)", ha="right", va="bottom", fontsize=6.5, color=TINTA)
bx.set_yscale("log")
bx.set_xlim(0, len(x) + 1)
bx.set_ylim(1, 3000)
bx.set_xlabel("Classe do IP102 (ordenada pelo número de imagens)", fontsize=7)
bx.set_ylabel("Número de imagens (escala log)", fontsize=7)
bx.tick_params(labelsize=6.5, length=2.5)
bx.set_xticks([1, 20, 40, 60, 80, 97])
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
bx.legend(fontsize=6.4, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.02))
salvar(fig, "rfs_ip102")
