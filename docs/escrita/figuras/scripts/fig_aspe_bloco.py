from estilo import *

fig = plt.figure(figsize=(6.4, 3.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 61); ax.axis("off")

yc, h = 44, 10          # linha principal
def bloco(x, w, t, sub, cores=AZUL, fs=7.4, peso="normal", estilo_fonte="normal"):
    return caixa(ax, x, yc - h / 2, w, h, t, cores, fs=fs, sub=sub, sub_fs=6.4, peso=peso, estilo_fonte=estilo_fonte)

caixa(ax, 0.5, yc - 3.5, 7, 7, "X", CINZA_CX, fs=9, peso="bold")
ax.text(4, yc - 5.6, "C×H×W", ha="center", va="top", fontsize=6.4, color=CINZA)
U1 = bloco(10, 13.5, "Conv 1×1\n+ BN + SiLU", "C → C/2")
U2 = bloco(26, 13.5, "Conv 3×3\n+ BN + SiLU", r"$U_2$: C/2×H×W")
CA = bloco(42.5, 13.5, "Coordinate\nAttention", r"$U_3 = U_2 \odot A_h \odot A_w$", ROXO, peso="bold", estilo_fonte="italic")
SE = bloco(59, 13.5, "Squeeze-and-\nExcitation", r"$U_4 = U_3 \odot s$", ROXO, peso="bold", estilo_fonte="italic")
CC = bloco(75.5, 8.5, "Concat", r"$[U_4\,;\,X]$", CINZA_CX)
FN = bloco(86.5, 12.5, "Conv 1×1\n+ BN + SiLU", "C/2 + C → C")
for a, b in [((7.5, yc), (10, yc)), ((23.5, yc), (26, yc)), ((39.5, yc), (42.5, yc)), ((56, yc), (59, yc)),
             ((72.5, yc), (75.5, yc)), ((84, yc), (86.5, yc))]:
    seta(ax, a, b, ms=6)
# identidade X até o Concat
linha(ax, [4, 4], [yc + 3.5, 56]); linha(ax, [4, 79.75], [56, 56])
seta(ax, (79.75, 56), (79.75, yc + h / 2), ms=6)
ax.text(42, 57.2, "identidade (X)", ha="center", va="bottom", fontsize=6.6, color=CINZA)
# saída Y
seta(ax, (92.75, yc - h / 2), (92.75, yc - h / 2 - 3.8), ms=6)
ax.text(92.75, yc - h / 2 - 4.4, "Y: C×H×W", ha="center", va="top", fontsize=7.2, weight="bold", color=TINTA)

ax.text(0.8, 27.5, r"$\odot$: produto elemento a" + "\nelemento (eixos de\ntamanho 1 são\nreplicados)",
        ha="left", va="center", fontsize=6.2, color=CINZA, linespacing=1.25)

# ---------------- detalhe da Coordinate Attention ----------------
dx, dy, dw, dh = 15, 1.0, 43.5, 26
caixa(ax, dx, dy, dw, dh, "", ("#FBF8FD", ROXO[1]), lw=0.7, ls=(0, (3, 2)), raio=0.01, z=0)
ax.text(dx + 1.2, dy + dh - 2.2, "Coordinate Attention", ha="left", va="center", fontsize=7.3, weight="bold", style="italic", color=ROXO[1])
linha(ax, [CA[0] - 3, dx + dw - 9], [yc - h / 2, dy + dh], cor=ROXO[1], lw=0.7, ls=(0, (3, 2)))
linha(ax, [CA[0] + 3, dx + dw - 0.5], [yc - h / 2, dy + dh], cor=ROXO[1], lw=0.7, ls=(0, (3, 2)))
caixa(ax, dx + 1.5, dy + 12.5, 12, 7.0, "Média ao\nlongo de W", CINZA_CX, fs=6.5)
caixa(ax, dx + 1.5, dy + 2.8, 12, 7.0, "Média ao\nlongo de H", CINZA_CX, fs=6.5)
ax.text(dx + 7.5, dy + 11.9, "C/2×H×1", ha="center", va="top", fontsize=5.9, color=CINZA)
ax.text(dx + 7.5, dy + 2.2, "C/2×1×W", ha="center", va="top", fontsize=5.9, color=CINZA)
caixa(ax, dx + 16.0, dy + 5.4, 11.5, 11.5, "Concat +\nConv 1×1\n+ BN +\n" + r"$\mathit{h}$-$\mathit{swish}$", CINZA_CX, fs=6.4)
caixa(ax, dx + 30.5, dy + 12.5, 11.5, 7.0, "Conv 1×1\n+ sigmoide", ROXO, fs=6.4)
caixa(ax, dx + 30.5, dy + 2.8, 11.5, 7.0, "Conv 1×1\n+ sigmoide", ROXO, fs=6.4)
ax.text(dx + 36.25, dy + 11.9, r"$A_h$: C/2×H×1", ha="center", va="top", fontsize=5.9, color=ROXO[1])
ax.text(dx + 36.25, dy + 2.2, r"$A_w$: C/2×1×W", ha="center", va="top", fontsize=5.9, color=ROXO[1])
seta(ax, (dx + 13.5, dy + 16.0), (dx + 16.0, dy + 13.8), ms=5)
seta(ax, (dx + 13.5, dy + 6.3), (dx + 16.0, dy + 8.5), ms=5)
seta(ax, (dx + 27.5, dy + 13.8), (dx + 30.5, dy + 16.0), ms=5)
seta(ax, (dx + 27.5, dy + 8.5), (dx + 30.5, dy + 6.3), ms=5)

# ---------------- detalhe da SE ----------------
sx, sy, sw, sh = 61.5, 1.0, 37.5, 26
caixa(ax, sx, sy, sw, sh, "", ("#FBF8FD", ROXO[1]), lw=0.7, ls=(0, (3, 2)), raio=0.01, z=0)
ax.text(sx + 1.2, sy + sh - 2.2, "Squeeze-and-Excitation", ha="left", va="center", fontsize=7.3, weight="bold", style="italic", color=ROXO[1])
linha(ax, [SE[0] - 3, sx + 4], [yc - h / 2, sy + sh], cor=ROXO[1], lw=0.7, ls=(0, (3, 2)))
linha(ax, [SE[0] + 3, sx + 16], [yc - h / 2, sy + sh], cor=ROXO[1], lw=0.7, ls=(0, (3, 2)))
caixa(ax, sx + 1.5, sy + 6.5, 10, 9, "Média\nglobal", CINZA_CX, fs=6.5)
ax.text(sx + 6.5, sy + 5.8, "C/2×1×1", ha="center", va="top", fontsize=5.9, color=CINZA)
caixa(ax, sx + 14, sy + 6.5, 11, 9, "FC → ReLU\n→ FC", CINZA_CX, fs=6.4)
caixa(ax, sx + 27.5, sy + 6.5, 8.5, 9, "sigmoide", ROXO, fs=6.4)
ax.text(sx + 31.75, sy + 5.8, r"$s$: C/2×1×1", ha="center", va="top", fontsize=5.9, color=ROXO[1])
seta(ax, (sx + 11.5, sy + 11), (sx + 14, sy + 11), ms=5)
seta(ax, (sx + 25, sy + 11), (sx + 27.5, sy + 11), ms=5)

salvar(fig, "aspe_bloco")
