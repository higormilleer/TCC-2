"""Visão geral da metodologia (Figura do Capítulo 4, Material e Métodos)."""
from estilo import *

fig = plt.figure(figsize=(6.5, 2.3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 35); ax.axis("off")

VERM = ("#F8DEDB", "#A93226")
etapas = [
    ("1. Aquisição dos dados", "AgroPest-12 (Kaggle) e\nsubconjunto de detecção\ndo IP102", AZUL),
    ("2. Auditoria e limpeza", "conferência dos rótulos e\nbusca de quase duplicatas\n(" + r"$\mathit{hash}$" + " perceptual)", AZUL),
    ("3. Divisão sem vazamento", "treino, validação e teste\npor grupos de imagens,\ncom estratificação", AZUL),
    ("4. Protocolo único", "AdamW, até 300 épocas,\naumento de dados fixo;\ntaxa inicial por estudo curto", VERDE),
    ("5. Experimentos", "arquiteturas, ASPE residual,\nsementes, ablação do aumento\ne técnicas para raras", LARANJA),
    ("6. Decisões na validação", r"$\mathit{fitness}$ do melhor $\mathit{checkpoint}$;" + "\nhipóteses e limiares\nregistrados antes", ROXO),
    ("7. Teste único", "lista fixa de modelos,\navaliada uma só vez,\nsem nenhum ajuste depois", VERM),
    ("8. Análises complementares", "perdas, variante sem zero\ne leitura alternativa do ASPE,\nsó na validação", CINZA_CX),
]
w, h = 22.8, 12.6
xs = [0.6, 25.8, 51.0, 76.2]
y_top, y_bot = 20.6, 1.2
pos = [(xs[0], y_top), (xs[1], y_top), (xs[2], y_top), (xs[3], y_top),
       (xs[3], y_bot), (xs[2], y_bot), (xs[1], y_bot), (xs[0], y_bot)]
for (t, s, cor), (x, y) in zip(etapas, pos):
    caixa(ax, x, y, w, h, "", cor, lw=0.9)
    ax.text(x + w / 2, y + h - 2.3, t, ha="center", va="center", fontsize=6.6, weight="bold", color=TINTA)
    ax.text(x + w / 2, y + h / 2 - 1.6, s, ha="center", va="center", fontsize=6.1, color=CINZA, linespacing=1.22)
for i in range(3):
    seta(ax, (xs[i] + w + 0.3, y_top + h / 2), (xs[i + 1] - 0.3, y_top + h / 2), ms=6)
seta(ax, (xs[3] + w / 2, y_top - 0.3), (xs[3] + w / 2, y_bot + h + 0.3), ms=6)
for i in (3, 2, 1):
    seta(ax, (xs[i] - 0.3, y_bot + h / 2), (xs[i - 1] + w + 0.3, y_bot + h / 2), ms=6)
salvar(fig, "Metodologia_fluxo")  # mesmo nome do arquivo no Overleaf
