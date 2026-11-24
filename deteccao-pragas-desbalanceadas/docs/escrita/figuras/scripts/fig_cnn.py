"""Figura da CNN (substitui a figura baseada na NVIDIA). Os mapas de características são calculados de verdade
com filtros fixos (bordas orientadas, blobs e oposição de cores) sobre uma imagem do AgroPest-12; as
probabilidades da saída são ilustrativas."""
import os
import cv2
import numpy as np
from matplotlib.patches import Rectangle
from estilo import *

BASE = os.path.join(RAIZ, "data", "processed", "agropest12_plano", "images", "train")
nome = [f for f in os.listdir(BASE) if f.startswith("Weevil-438-_jpg")][0]
img = cv2.cvtColor(cv2.imread(os.path.join(BASE, nome)), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
gray = img.mean(axis=2)

def relu(a):
    return np.maximum(a, 0)

def norm(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.3)
    a = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
    return 0.08 + 0.92 * a ** 0.7

k = {
    "sx": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
    "sy": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
    "d1": np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], np.float32),
    "d2": np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], np.float32),
}
f = lambda a, kk: cv2.filter2D(a, -1, kk, borderType=cv2.BORDER_REFLECT)
log = cv2.GaussianBlur(gray, (0, 0), 2.0)
log = -cv2.Laplacian(log, cv2.CV_32F, ksize=3)
bordas = np.sqrt(f(gray, k["sx"]) ** 2 + f(gray, k["sy"]) ** 2)
conv1 = [
    bordas, relu(f(gray, k["sx"])), relu(-f(gray, k["sx"])), relu(f(gray, k["sy"])), relu(f(gray, k["d1"])),
    relu(f(gray, k["d2"])), relu(-log), relu(img[..., 0] - img[..., 1] + 0.05), relu(gray.mean() - cv2.GaussianBlur(gray, (0, 0), 3)),
]
pool = lambda a: a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).max(axis=(1, 3))
pool1 = [pool(m) for m in conv1]
conv2 = []
ks = list(k.values())
for i in range(16):
    a = pool1[i % 8]; b = pool1[(i + 3) % 8]
    conv2.append(relu(f(a, ks[i % 4]) + 0.5 * f(b, ks[(i + 1) % 4])))
conv2[0] = cv2.GaussianBlur(pool1[0], (0, 0), 1.2) * (pool1[5] > np.percentile(pool1[5], 60))  # partes do corpo (bordas internas)
pool2 = [pool(m) for m in conv2]

def ordenar(maps):  # mapa de maior contraste na frente
    return sorted(maps, key=lambda m: -np.std(norm(m)))

fig = plt.figure(figsize=(6.5, 2.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 134); ax.set_ylim(0, 44); ax.axis("off")

def pilha(x, y, s, mapas, n, off, rgb=False, cmap="gray"):
    """Desenha n planos de lado s, do fundo para a frente, com deslocamento off para cima/direita."""
    for j in reversed(range(n)):
        xj, yj = x + j * off, y + j * off
        z = 10 + (n - j)
        if rgb:
            if j == 0:
                ax.imshow(img, extent=[xj, xj + s, yj, yj + s], aspect="auto", zorder=z, interpolation="bilinear")
            else:
                cm = ["Reds", "Greens", "Blues"][j]
                ax.imshow(img[..., j], extent=[xj, xj + s, yj, yj + s], aspect="auto", cmap=cm, vmin=0, vmax=1.3,
                          zorder=z, interpolation="bilinear")
        else:
            ax.imshow(norm(mapas[j % len(mapas)]), extent=[xj, xj + s, yj, yj + s], aspect="auto", cmap=cmap,
                      vmin=0, vmax=1, zorder=z, interpolation="nearest")
        ax.add_patch(Rectangle((xj, yj), s, s, fill=False, ec="#3A4048", lw=0.45, zorder=z + 0.5))
    return x + (n - 1) * off + s  # borda direita

y0 = 12
x = 1.0
r = pilha(x, y0, 13, None, 3, 1.3, rgb=True)
ax.text(x + 7.6, y0 - 1.5, "Entrada\n128×128×3", ha="center", va="top", fontsize=6.6, color=TINTA, linespacing=1.15)

def op(xa, xb, texto):
    seta(ax, (xa + 0.6, y0 + 6.5), (xb - 0.6, y0 + 6.5), ms=5.5, lw=0.8)
    ax.text((xa + xb) / 2, y0 + 23.5, texto, ha="center", va="bottom", fontsize=6.2, color=CINZA, linespacing=1.15)

x2 = r + 6.5
op(r, x2, "Convolução\n+ ReLU")
r2 = pilha(x2, y0, 13, conv1, 8, 0.75)
ax.text(x2 + 9.5, y0 - 1.5, "Mapas de\ncaracterísticas\n128×128×8", ha="center", va="top", fontsize=6.6, linespacing=1.15)

x3 = r2 + 6.5
op(r2, x3, "Subamostragem\n(max pooling)")
r3 = pilha(x3, y0, 8.5, pool1, 8, 0.6)
ax.text(x3 + 6.3, y0 - 1.5, "64×64×8", ha="center", va="top", fontsize=6.6)

x4 = r3 + 6.5
op(r3, x4, "Convolução\n+ ReLU")
r4 = pilha(x4, y0, 8.5, conv2, 16, 0.42)
ax.text(x4 + 7.4, y0 - 1.5, "64×64×16", ha="center", va="top", fontsize=6.6)

x5 = r4 + 6.0
op(r4, x5, "Subamostragem")
r5 = pilha(x5, y0, 5.0, pool2, 16, 0.3)
ax.text(x5 + 4.8, y0 - 1.5, "32×32×16", ha="center", va="top", fontsize=6.6)

# camada totalmente conectada
xf = r5 + 6.0
ys = np.linspace(y0 + 0.5, y0 + 18.5, 8)
for yy in ys:
    for yb in (y0 + 1, y0 + 4, y0 + 7):
        linha(ax, [r5, xf - 0.8], [yb, yy], cor="#B9BEC5", lw=0.3, z=0)
for yy in ys:
    ax.plot([xf], [yy], "o", ms=3.4, mfc="white", mec=TINTA, mew=0.6, zorder=5)
ax.text(xf, y0 - 1.5, "Camada\ntotalmente\nconectada", ha="center", va="top", fontsize=6.6, color=TINTA, linespacing=1.15)

# saída softmax (probabilidades ilustrativas)
classes = ["Formigas", "Abelhas", "Besouros", "Lagartas", "Minhocas", "Tesourinhas", "Gafanhotos", "Mariposas",
           "Lesmas", "Caracóis", "Vespas", "Gorgulhos"]
p = np.array([0.006, 0.004, 0.072, 0.008, 0.003, 0.011, 0.012, 0.007, 0.004, 0.006, 0.005, 0.862])
p = p / p.sum()
xb0, larg = xf + 16.5, 9.0
yb = np.linspace(y0 + 18.6, y0 - 9.2, 12)
seta(ax, (xf + 1.4, y0 + 6.5), (xf + 4.4, y0 + 6.5), ms=5.5, lw=0.8)
for c, pp, yy in zip(classes, p, yb):
    destaque = pp == p.max()
    ax.text(xb0 - 0.5, yy, c, ha="right", va="center", fontsize=5.9, color=TINTA, weight="bold" if destaque else "normal")
    ax.add_patch(Rectangle((xb0, yy - 0.85), larg, 1.7, fc="#EEF0F2", ec="none", zorder=2))
    ax.add_patch(Rectangle((xb0, yy - 0.85), larg * pp, 1.7, fc=VERMELHO if destaque else AZUL[1], ec="none", zorder=3))
    if destaque:
        ax.text(xb0 + larg + 0.6, yy, f"{pp:.2f}".replace(".", ","), ha="left", va="center", fontsize=5.9,
                weight="bold", color=VERMELHO)
ax.text(xb0 - 3, y0 + 23.5, "Softmax: probabilidade\nde cada classe", ha="center", va="bottom", fontsize=6.2,
        color=CINZA, linespacing=1.15)

salvar(fig, "cnn_pragas")
