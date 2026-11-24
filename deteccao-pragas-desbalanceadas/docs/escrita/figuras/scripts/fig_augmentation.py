"""Exemplos das transformações de aumento de dados do protocolo (A01) e das técnicas para classes minoritárias,
aplicadas a imagens reais do AgroPest-12, com as caixas transformadas junto com a imagem."""
import os
import cv2
import numpy as np
from matplotlib.patches import Rectangle
from estilo import *

BASE = os.path.join(RAIZ, "data", "processed", "agropest12_plano")
NOMES = ["Formigas", "Abelhas", "Besouros", "Lagartas", "Minhocas", "Tesourinhas", "Gafanhotos", "Mariposas",
         "Lesmas", "Caracóis", "Vespas", "Gorgulhos"]
CORES = {0: "#E67E22", 1: "#F1C40F", 3: "#E74C3C", 4: "#9B59B6", 5: "#1ABC9C", 6: "#2ECC71", 8: "#3498DB",
         9: "#E84393", 10: "#F39C12", 11: "#8E5A2B"}
S = 320

def carregar(stem):
    im_dir = os.path.join(BASE, "images", "train")
    fn = [f for f in os.listdir(im_dir) if f.startswith(stem)][0]
    img = cv2.cvtColor(cv2.imread(os.path.join(im_dir, fn)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
    caixas = []
    for l in open(os.path.join(BASE, "labels", "train", stem + ".txt")):
        p = l.split()
        if len(p) >= 5:
            c, x, y, w, h = int(p[0]), *map(float, p[1:5])
            caixas.append([c, (x - w / 2) * S, (y - h / 2) * S, (x + w / 2) * S, (y + h / 2) * S])
    return img, np.array(caixas, dtype=np.float32)

def clip(cx, W=S, H=S, min_area=0.15):
    out = []
    for c, x1, y1, x2, y2 in cx:
        a0 = max(1.0, (x2 - x1) * (y2 - y1))
        nx1, ny1, nx2, ny2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if nx2 - nx1 > 4 and ny2 - ny1 > 4 and (nx2 - nx1) * (ny2 - ny1) / a0 > min_area:
            out.append([c, nx1, ny1, nx2, ny2])
    return np.array(out, dtype=np.float32).reshape(-1, 5)

def hflip(img, cx):
    cx = cx.copy(); x1 = S - cx[:, 3]; x2 = S - cx[:, 1]; cx[:, 1], cx[:, 3] = x1, x2
    return img[:, ::-1].copy(), cx

def vflip(img, cx):
    cx = cx.copy(); y1 = S - cx[:, 4]; y2 = S - cx[:, 2]; cx[:, 2], cx[:, 4] = y1, y2
    return img[::-1].copy(), cx

def hsv(img, cx, r=(1.015, 0.45, 1.3)):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    h = ((h.astype(np.float32) * r[0]) % 180).astype(np.uint8)
    s = np.clip(s.astype(np.float32) * r[1], 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.float32) * r[2], 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB), cx.copy()

def afim(img, cx, ang=12, esc=1.15, tx=0.06, ty=-0.05):
    M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, esc)
    M[0, 2] += tx * S; M[1, 2] += ty * S
    out = cv2.warpAffine(img, M, (S, S), borderValue=(114, 114, 114))
    novas = []
    for c, x1, y1, x2, y2 in cx:
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]) @ M.T
        novas.append([c, pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    return out, clip(np.array(novas, dtype=np.float32))

def mosaic(itens, xc=0.55, yc=0.47):
    T = 2 * S
    can = np.full((T, T, 3), 114, np.uint8)
    cxs, cys = int(xc * T), int(yc * T)
    todas = []
    for k, (img, cx) in enumerate(itens):
        if k == 0:   x1a, y1a, x2a, y2a = max(cxs - S, 0), max(cys - S, 0), cxs, cys; x1b, y1b = S - (x2a - x1a), S - (y2a - y1a)
        elif k == 1: x1a, y1a, x2a, y2a = cxs, max(cys - S, 0), min(cxs + S, T), cys; x1b, y1b = 0, S - (y2a - y1a)
        elif k == 2: x1a, y1a, x2a, y2a = max(cxs - S, 0), cys, cxs, min(T, cys + S); x1b, y1b = S - (x2a - x1a), 0
        else:        x1a, y1a, x2a, y2a = cxs, cys, min(cxs + S, T), min(T, cys + S); x1b, y1b = 0, 0
        can[y1a:y2a, x1a:x2a] = img[y1b:y1b + (y2a - y1a), x1b:x1b + (x2a - x1a)]
        dx, dy = x1a - x1b, y1a - y1b
        c2 = cx.copy(); c2[:, [1, 3]] += dx; c2[:, [2, 4]] += dy
        todas.append(clip(c2, T, T))
    todas = np.concatenate(todas)
    out = cv2.resize(can, (S, S), interpolation=cv2.INTER_AREA)
    todas[:, 1:] *= S / T
    return out, todas

def mixup(a, b, r=0.5):
    (ia, ca), (ib, cb) = a, b
    return (ia.astype(np.float32) * r + ib.astype(np.float32) * (1 - r)).astype(np.uint8), np.concatenate([ca, cb])

def copy_paste(alvo, fonte, pos=(0.03, 0.05), esc=0.5):
    (it, ct), (iff, cf) = alvo, fonte
    c, x1, y1, x2, y2 = cf[0]
    m = 0.04 * S
    x1, y1, x2, y2 = int(max(0, x1 - m)), int(max(0, y1 - m)), int(min(S, x2 + m)), int(min(S, y2 + m))
    rec = iff[y1:y2, x1:x2]
    rec = cv2.resize(rec, (int(rec.shape[1] * esc), int(rec.shape[0] * esc)), interpolation=cv2.INTER_AREA)
    h, w = rec.shape[:2]
    px, py = int(pos[0] * S), int(pos[1] * S)
    h = min(h, S - py); w = min(w, S - px); rec = rec[:h, :w]
    mask = np.zeros((h, w), np.float32); b = max(2, int(0.08 * min(h, w)))
    mask[b:-b, b:-b] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), b / 2)[..., None]
    out = it.astype(np.float32).copy()
    out[py:py + h, px:px + w] = rec * mask + out[py:py + h, px:px + w] * (1 - mask)
    mm = m * esc
    nova = np.array([[c, px + mm, py + mm, px + w - mm, py + h - mm]], np.float32)
    return out.astype(np.uint8), np.concatenate([ct, nova])

base = carregar("bees-346-_jpg.rf.145642fa340fdef0bb871d18251da22b")
paineis = [
    ("Original", base),
    ("Espelhamento horizontal", hflip(*base)),
    ("Espelhamento vertical", vflip(*base)),
    ("Variação de cor (HSV)", hsv(*base)),
    ("Rotação, escala e translação", afim(*base)),
    ("Mosaic", mosaic([carregar("catterpillar-438-_jpg.rf.2e1ff76818e0dc97e1598cae8597e7f1"),
                       carregar("grasshopper-92-_jpg.rf.ef7f8ef3844a2d06d1f68dba75fea696"),
                       carregar("wasp-926-_jpg.rf.4d86185e5c9dba832d14231129fe8622"),
                       carregar("ants-429-_jpg.rf.227c883596af5ed08cc8424939bb2541")])),
    ("MixUp", mixup(carregar("earthworms-87-_jpg.rf.8a20aedb5416969c4809282cc5b66237"),
                    carregar("snail-147-_jpg.rf.27412ac4d21ff1e6d15bc50c60df355c"))),
    ("Copy-Paste", copy_paste(carregar("Weevil-438-_jpg.rf.97afc57d399cd9b11b4a58a33de1b690"),
                              carregar("slug-351-_jpg.rf.2d3f1077309503ff36a4813cebbc37c9"))),
]

fig, axs = plt.subplots(2, 4, figsize=(6.5, 3.55))
fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01, wspace=0.04, hspace=0.16)
for ax, (titulo, (img, cx)) in zip(axs.ravel(), paineis):
    ax.imshow(img); ax.set_xlim(0, S); ax.set_ylim(S, 0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#8A9099")
    for c, x1, y1, x2, y2 in cx:
        cor = CORES.get(int(c), "#FFFFFF")
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=cor, lw=1.1))
        ty = y1 - 2 if y1 > 22 else y1 + 2
        ax.text(x1 + 1, ty, NOMES[int(c)], fontsize=5.2, color="white", va="bottom" if y1 > 22 else "top",
                ha="left", bbox=dict(boxstyle="square,pad=0.12", fc=cor, ec="none"))
    ax.set_title(titulo, fontsize=6.9, weight="bold" if titulo == "Original" else "normal", pad=2.5)
salvar(fig, "augmentation_exemplos")
