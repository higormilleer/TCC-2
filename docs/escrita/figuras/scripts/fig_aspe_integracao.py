from estilo import *

fig = plt.figure(figsize=(6.4, 3.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")

def soma(x, y):
    ax.plot([x], [y], marker="o", ms=13, mfc="white", mec=TINTA, mew=0.9, zorder=4)
    ax.text(x, y - 0.1, "+", ha="center", va="center", fontsize=11, color=TINTA, zorder=5)

# ---------------- (a) original ----------------
ax.text(24, 55.5, "(a) ASPE original (em série)", ha="center", va="center", fontsize=8.6, weight="bold")
cx = 24
ax.text(cx, 50.5, "x", ha="center", va="center", fontsize=8.5, style="italic")
seta(ax, (cx, 49), (cx, 45), ms=6)
caixa(ax, cx - 12, 37, 24, 8, "C2f ou C3k2", AZUL, fs=8, peso="bold", sub="pesos pré-treinados", sub_fs=6.6)
seta(ax, (cx, 37), (cx, 33), ms=6)
caixa(ax, cx - 12, 25, 24, 8, "ASPE", ROXO, fs=8, peso="bold", sub="pesos aleatórios", sub_fs=6.6)
seta(ax, (cx, 25), (cx, 20.5), ms=6)
ax.text(cx, 18.8, "saída = ASPE(bloco(x))", ha="center", va="center", fontsize=7.4)
ax.text(cx, 9, "No início do ajuste fino, a saída do bloco é\npraticamente aleatória: as camadas seguintes,\npré-treinadas, recebem um sinal diferente\ndaquele para o qual foram ajustadas.",
        ha="center", va="center", fontsize=6.6, color=CINZA, linespacing=1.3)

linha(ax, [50, 50], [3, 56], cor="#C9CED4", lw=0.6)

# ---------------- (b) residual ----------------
ax.text(76, 55.5, "(b) ASPE residual (proposto)", ha="center", va="center", fontsize=8.6, weight="bold")
cx = 80
ax.text(cx, 50.5, "x", ha="center", va="center", fontsize=8.5, style="italic")
seta(ax, (cx, 49), (cx, 45), ms=6)
caixa(ax, cx - 12, 37, 24, 8, "C2f ou C3k2", AZUL, fs=8, peso="bold", sub="pesos pré-treinados", sub_fs=6.6)
linha(ax, [cx, cx], [37, 34.5])
ax.plot([cx], [34.5], "o", ms=2.8, color=TINTA, zorder=3)
ax.text(cx + 1.0, 35.2, "y", ha="left", va="center", fontsize=8.5, style="italic")
# ramo ASPE
seta(ax, (cx, 34.5), (cx, 31.5), ms=6)
caixa(ax, cx - 12, 22, 24, 9.5, "ASPE", ROXO, fs=8, peso="bold", sub=r"última BN: $\gamma = \beta = 0$", sub_fs=6.6)
seta(ax, (cx, 22), (cx, 18.6), ms=6)
# atalho identidade
xs = 60.5
linha(ax, [cx, xs], [34.5, 34.5]); linha(ax, [xs, xs], [34.5, 16.6])
seta(ax, (xs, 16.6), (cx - 1.7, 16.6), ms=6)
ax.text(xs - 0.8, 26, "atalho (identidade)", ha="right", va="center", fontsize=6.4, color=CINZA, rotation=90)
soma(cx, 16.6)
seta(ax, (cx, 14.9), (cx, 11.6), ms=6)
ax.text(cx, 10.0, "saída = y + ASPE(y)", ha="center", va="center", fontsize=7.4)
ax.text(76, 3.6, "No início, ASPE(y) = 0 e a saída é exatamente y: a rede\ncomeça idêntica à pré-treinada e o ASPE contribui aos poucos.",
        ha="center", va="center", fontsize=6.6, color=CINZA, linespacing=1.3)

salvar(fig, "aspe_integracao")
