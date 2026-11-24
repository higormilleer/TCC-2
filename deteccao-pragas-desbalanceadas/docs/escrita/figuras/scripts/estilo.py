"""Estilo comum das figuras do TCC (fonte, cores, caixas, setas, salvamento)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# raiz do repositorio (este arquivo fica em docs/escrita/figuras/scripts)
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT = os.path.join(RAIZ, "docs", "escrita", "figuras")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
})

# paleta (boa em impressão e para daltônicos)
TINTA = "#1F2328"      # texto e setas
CINZA = "#5B6470"      # texto secundário
AZUL = ("#DCEAF7", "#2F5D8A")      # backbone
VERDE = ("#E2F0DE", "#3B7A39")     # neck
LARANJA = ("#FBE6D2", "#B05F1A")   # cabeça
ROXO = ("#ECDDF5", "#6E3596")      # ASPE
CINZA_CX = ("#EEF0F2", "#59616B")  # neutro
VERMELHO = "#C0392B"

def caixa(ax, x, y, w, h, texto="", cores=CINZA_CX, fs=8.5, peso="normal", sub=None, sub_fs=7, raio=0.02,
          lw=0.9, ls="-", cor_texto=TINTA, alpha=1.0, z=2):
    """Caixa arredondada com (x, y) = canto inferior esquerdo. Devolve o centro."""
    fc, ec = cores
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={raio}", fc=fc, ec=ec, lw=lw, ls=ls,
                       alpha=alpha, zorder=z)
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + h * 0.16, texto, ha="center", va="center", fontsize=fs, weight=peso, color=cor_texto, zorder=z + 1)
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center", fontsize=sub_fs, color=CINZA, zorder=z + 1)
    elif texto:
        ax.text(cx, cy, texto, ha="center", va="center", fontsize=fs, weight=peso, color=cor_texto, zorder=z + 1)
    return cx, cy

def seta(ax, p0, p1, cor=TINTA, lw=0.9, estilo="-|>", ms=7, ls="-", conn="arc3", z=1, shrinkA=0, shrinkB=0):
    a = FancyArrowPatch(p0, p1, arrowstyle=estilo, mutation_scale=ms, color=cor, lw=lw, ls=ls,
                        connectionstyle=conn, zorder=z, shrinkA=shrinkA, shrinkB=shrinkB)
    ax.add_patch(a)
    return a

def linha(ax, xs, ys, cor=TINTA, lw=0.9, ls="-", z=1):
    ax.plot(xs, ys, color=cor, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")

def salvar(fig, nome):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{nome}.{ext}"), bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print("salvo:", os.path.join(OUT, nome + ".pdf"))
