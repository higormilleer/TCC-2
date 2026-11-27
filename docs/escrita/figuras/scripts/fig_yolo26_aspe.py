from estilo import *

fig = plt.figure(figsize=(6.4, 5.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(3, 84); ax.axis("off")

# ---------------- backbone ----------------
bx, bw, bh, passo, topo = 12, 22, 4.4, 6.2, 74
linhas = [
    ("Entrada", "640×640×3", CINZA_CX, False),
    ("Conv (s = 2)", "320×320×32", AZUL, False),
    ("Conv (s = 2)", "160×160×64", AZUL, False),
    ("C3k2", "160×160×128", AZUL, True),
    ("Conv (s = 2)", "80×80×128", AZUL, False),
    ("C3k2", "80×80×256", AZUL, True),
    ("Conv (s = 2)", "40×40×256", AZUL, False),
    ("C3k2", "40×40×256", AZUL, True),
    ("Conv (s = 2)", "20×20×512", AZUL, False),
    ("C3k2", "20×20×512", AZUL, True),
    ("SPPF", "20×20×512", AZUL, False),
    ("C2PSA (atenção)", "20×20×512", AZUL, False),
]
cy = {}
for r, (t, s, cor, aspe) in enumerate(linhas):
    y = topo - r * passo
    caixa(ax, bx, y, bw, bh, t, cor, fs=8, peso="bold" if t.startswith("C3k2") else "normal", sub=s, sub_fs=6.8)
    cy[r] = y + bh / 2
    if r > 0:
        seta(ax, (bx + bw / 2, y + passo), (bx + bw / 2, y + bh), ms=6)
    if aspe:
        caixa(ax, 1.5, y + 0.6, 7.5, bh - 1.2, "ASPE", ROXO, fs=7.5, peso="bold", cor_texto=ROXO[1])
        linha(ax, [9.0, bx], [cy[r], cy[r]], cor=ROXO[1], lw=1.0)
ax.text(bx + bw / 2, 81, "Backbone", ha="center", va="center", fontsize=9.5, weight="bold", style="italic",
        color=AZUL[1])

# ---------------- neck ----------------
nw, nh = 15, 5.0
xA = 44            # coluna FPN
xC = 65            # coluna PAN
yP3, yP4, yP5 = cy[5], cy[7], cy[11]
A = caixa(ax, xA, yP4 - nh / 2, nw, nh, "Concat + C3k2", VERDE, fs=7.5, sub="40×40×256", sub_fs=6.6)
B = caixa(ax, xA, yP3 - nh / 2, nw, nh, "Concat + C3k2", VERDE, fs=7.5, sub="80×80×128", sub_fs=6.6)
C = caixa(ax, xC, yP4 - nh / 2, nw, nh, "Concat + C3k2", VERDE, fs=7.5, sub="40×40×256", sub_fs=6.6)
D = caixa(ax, xC, yP5 - nh / 2, nw, nh, "Concat + C3k2", VERDE, fs=7.5, sub="20×20×512", sub_fs=6.6)
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
_t1 = TextArea("Neck", textprops=dict(fontsize=9.5, weight="bold", style="italic", color=VERDE[1]))
_t2 = TextArea("(FPN + PAN)", textprops=dict(fontsize=9.5, weight="bold", color=VERDE[1]))
ax.add_artist(AnnotationBbox(HPacker(children=[_t1, _t2], align="baseline", pad=0, sep=3.2),
                             ((xA + xC + nw) / 2, 81), frameon=False, box_alignment=(0.5, 0.5)))

# derivações do backbone
seta(ax, (bx + bw, yP3), (xA, yP3), ms=6); ax.text(38.5, yP3 + 1.0, "P3", ha="center", fontsize=7, color=CINZA)
seta(ax, (bx + bw, yP4), (xA, yP4), ms=6); ax.text(38.5, yP4 + 1.0, "P4", ha="center", fontsize=7, color=CINZA)
xj = xA + nw / 2
linha(ax, [bx + bw, xC], [yP5, yP5]); ax.text(38.5, yP5 + 1.0, "P5", ha="center", fontsize=7, color=CINZA)
seta(ax, (xC - 3, yP5), (xC, yP5), ms=6)
ax.plot([xj], [yP5], "o", ms=2.8, color=TINTA, zorder=3)
# caminho FPN (profundo -> raso)
seta(ax, (xj, yP5), (xj, yP4 - nh / 2), ms=6)
ax.text(xj - 0.8, (yP5 + yP4) / 2, r"$\mathit{Upsample}$ ×2", ha="right", va="center", fontsize=6.6, color=CINZA)
seta(ax, (xj, yP4 + nh / 2), (xj, yP3 - nh / 2), ms=6)
ax.text(xj - 0.8, (yP4 + yP3) / 2, r"$\mathit{Upsample}$ ×2", ha="right", va="center", fontsize=6.6, color=CINZA)
# caminho PAN (raso -> profundo)
seta(ax, (xA + nw - 2.5, yP3 - nh / 2), (xC + 3.0, yP4 + nh / 2), ms=6)
ax.text(62.6, (yP3 + yP4) / 2 + 0.6, "Conv (s = 2)", ha="left", va="center", fontsize=6.6, color=CINZA)
seta(ax, (xA + nw, yP4), (xC, yP4), ms=6)
seta(ax, (xC + nw / 2, yP4 - nh / 2), (xC + nw / 2, yP5 + nh / 2), ms=6)
ax.text(xC + nw / 2 + 0.8, (yP4 + yP5) / 2, "Conv (s = 2)", ha="left", va="center", fontsize=6.6, color=CINZA, rotation=90)

# ---------------- cabeças ----------------
hx, hw, hh = 84, 15, 6.0
caixa(ax, hx, yP3 - hh / 2, hw, hh, "Cabeça P3", LARANJA, fs=7.8, peso="bold", sub="80×80: pequenos", sub_fs=6.5)
caixa(ax, hx, yP4 - hh / 2, hw, hh, "Cabeça P4", LARANJA, fs=7.8, peso="bold", sub="40×40: médios", sub_fs=6.5)
caixa(ax, hx, yP5 - hh / 2, hw, hh, "Cabeça P5", LARANJA, fs=7.8, peso="bold", sub="20×20: grandes", sub_fs=6.5)
seta(ax, (xA + nw, yP3), (hx, yP3), ms=6)
seta(ax, (xC + nw, yP4), (hx, yP4), ms=6)
seta(ax, (xC + nw, yP5), (hx, yP5), ms=6)
ax.text(hx + hw / 2, 81, "Cabeça", ha="center", va="center", fontsize=9.5, weight="bold", color=LARANJA[1])

# ---------------- legenda ----------------
lx, ly = 44, 58
caixa(ax, lx, ly, 55, 20, "", ("#FFFFFF", "#B8BEC6"), lw=0.6)
caixa(ax, lx + 2, ly + 14.3, 7.5, 3.6, "ASPE", ROXO, fs=7, peso="bold", cor_texto=ROXO[1])
ax.text(lx + 11, ly + 16.1, r"módulo inserido após os quatro C3k2 do $\mathit{backbone}$: em" + "\nsérie (ASPE original) ou como ramo (ASPE residual)",
        ha="left", va="center", fontsize=6.9, color=TINTA, linespacing=1.25)
ax.text(lx + 2, ly + 10.0, "s = 2: convolução com passo 2, que reduz a resolução\npela metade e define as escalas P3 (1/8), P4 (1/16) e P5 (1/32)",
        ha="left", va="center", fontsize=6.9, color=TINTA, linespacing=1.25)
ax.text(lx + 2, ly + 3.8, "Cada cabeça prevê caixas, classes e confianças por dois\nramos: um-para-muitos (seguido de NMS) e um-para-um",
        ha="left", va="center", fontsize=6.9, color=TINTA, linespacing=1.25)

salvar(fig, "yolo26s_aspe_arquitetura")
