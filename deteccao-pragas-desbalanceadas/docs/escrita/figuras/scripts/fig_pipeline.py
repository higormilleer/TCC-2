"""Pipeline de detecção (Figura 5) e exemplo de detecção (Figura 4) com o YOLO26s treinado (E03, seed 42, AgroPest-12),
rodando na CPU sobre imagens da VALIDAÇÃO (o teste não é usado)."""
import os, sys
import cv2
import numpy as np
from matplotlib.patches import Rectangle
from estilo import *

sys.path.insert(0, os.path.join(RAIZ, "src"))
from tcc.models.yolo_aspe import load_fp32  # noqa: E402

BASE = os.path.join(RAIZ, "data", "processed", "agropest12_plano")
PT = {"Ants": "Formigas", "Bees": "Abelhas", "Beetles": "Besouros", "Caterpillars": "Lagartas", "Earthworms": "Minhocas",
      "Earwigs": "Tesourinhas", "Grasshoppers": "Gafanhotos", "Moths": "Mariposas", "Slugs": "Lesmas", "Snails": "Caracóis",
      "Wasps": "Vespas", "Weevils": "Gorgulhos"}
y, ck = load_fp32(os.path.join(RAIZ, "runs", "plano", "plano_E03_s42", "weights", "best_fp32.pt"), device="cpu")
nomes = y.model.names

def caminho(stem):
    d = os.path.join(BASE, "images", "val")
    return os.path.join(d, [f for f in os.listdir(d) if f.startswith(stem)][0])

def gt(stem, W, H):
    out = []
    for l in open(os.path.join(BASE, "labels", "val", stem + ".txt")):
        c, x, yy, w, h = l.split()[:5]
        x, yy, w, h = float(x) * W, float(yy) * H, float(w) * W, float(h) * H
        out.append((int(c), x - w / 2, yy - h / 2, x + w / 2, yy + h / 2))
    return out

def preds(p, **kw):
    r = y.predict(p, device="cpu", verbose=False, **kw)[0]
    return [(int(c), float(s), *map(float, b)) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]

fmt = lambda v: f"{v:.2f}".replace(".", ",")
VERDE_DET = "#2ECC71"

# ======================= Figura do pipeline =======================
stem = "beetle-226-_jpg.rf.f705e7ee7c943079dce616343a462d95"
p = caminho(stem)
img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
cand = preds(p, conf=0.05, iou=0.95)
fin = preds(p)
print("pipeline:", W, H, "candidatas", len(cand), "finais", len(fin))

fig = plt.figure(figsize=(6.5, 1.68))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(4.5, 38); ax.axis("off")
L = 20.5   # lado dos painéis de imagem
y0 = 11.5
def painel_img(x, im, titulo, sub):
    ax.imshow(im, extent=[x, x + L, y0, y0 + L], aspect="auto", zorder=2)
    ax.add_patch(Rectangle((x, y0), L, L, fill=False, ec="#59616B", lw=0.5, zorder=3))
    ax.text(x + L / 2, y0 + L + 2.0, titulo, ha="center", va="bottom", fontsize=7.0, weight="bold")
    ax.text(x + L / 2, y0 - 1.3, sub, ha="center", va="top", fontsize=6.1, color=CINZA, linespacing=1.2)

def caixas(x, lst, com_rotulo, lw, alpha=1.0, cor=VERDE_DET):
    for it in lst:
        c, s, x1, y1, x2, y2 = it
        X1, X2 = x + x1 / W * L, x + x2 / W * L
        Y1, Y2 = y0 + L - y2 / H * L, y0 + L - y1 / H * L
        ax.add_patch(Rectangle((X1, Y1), X2 - X1, Y2 - Y1, fill=False, ec=cor, lw=lw, alpha=alpha, zorder=4))
        if com_rotulo:
            dentro = Y2 + 1.6 > y0 + L
            ax.text(X1 + 0.2, Y2 - 0.15 if dentro else Y2 + 0.15, fmt(s), fontsize=4.6, color="white",
                    va="top" if dentro else "bottom", ha="left", zorder=5, bbox=dict(boxstyle="square,pad=0.08", fc=cor, ec="none"))

xs = [1, 27, 53, 79, 105]
painel_img(xs[0], img, "Imagem de campo", "imagem original\nda lavoura")
# pré-processamento: planos RGB do tensor de entrada
im640 = cv2.resize(img, (640, 640)).astype(np.float32) / 255
for j, cm in reversed(list(enumerate(["Reds", "Greens", "Blues"]))):
    off = j * 1.4
    dado = im640 if j == 0 else im640[..., j]
    ax.imshow(dado, extent=[xs[1] + off, xs[1] + off + L - 2.8, y0 + off, y0 + off + L - 2.8], aspect="auto",
              cmap=None if j == 0 else cm, vmin=0, vmax=1.25, zorder=2 + (3 - j))
    ax.add_patch(Rectangle((xs[1] + off, y0 + off), L - 2.8, L - 2.8, fill=False, ec="#59616B", lw=0.45, zorder=2.5 + (3 - j)))
ax.text(xs[1] + L / 2, y0 + L + 2.0, "Pré-processamento", ha="center", va="bottom", fontsize=7.0, weight="bold")
ax.text(xs[1] + L / 2, y0 - 1.3, "640×640×3, valores em [0, 1]\n(no treino: aumento de dados)", ha="center", va="top",
        fontsize=6.1, color=CINZA, linespacing=1.2)
# modelo
mx = xs[2]
caixa(ax, mx + 1, y0 + 14.0, L - 2, 5.2, "Backbone", AZUL, fs=6.6)
caixa(ax, mx + 1, y0 + 7.6, L - 2, 5.2, "Neck", VERDE, fs=6.6)
caixa(ax, mx + 1, y0 + 1.2, L - 2, 5.2, "Cabeça", LARANJA, fs=6.6)
seta(ax, (mx + L / 2, y0 + 14.0), (mx + L / 2, y0 + 12.8), ms=4, lw=0.7)
seta(ax, (mx + L / 2, y0 + 7.6), (mx + L / 2, y0 + 6.4), ms=4, lw=0.7)
ax.text(mx + L / 2, y0 + L + 2.0, "Modelo de detecção", ha="center", va="bottom", fontsize=7.0, weight="bold")
ax.text(mx + L / 2, y0 - 1.3, "YOLO26s: caixa, classe e\nconfiança para cada posição", ha="center", va="top", fontsize=6.1,
        color=CINZA, linespacing=1.2)
# saída bruta
painel_img(xs[3], img, "Predições candidatas", f"{len(cand)} caixas com confiança\n≥ 0,05, antes da NMS")
caixas(xs[3], cand, False, 0.45, alpha=0.85, cor="#F1C40F")
# final
painel_img(xs[4], img, "Pós-processamento", f"limiar de 0,25 e NMS:\n{len(fin)} detecções finais")
caixas(xs[4], fin, True, 0.8)
for a, b in zip(xs[:-1], xs[1:]):
    seta(ax, (a + L + 0.6, y0 + L / 2), (b - 0.6, y0 + L / 2), ms=5.5, lw=0.8)
salvar(fig, "pipeline_deteccao")

# ======================= Exemplo de detecção =======================
stem = "ants-135-_jpg.rf.a193cf6478573752251b3088e35ebc29"
p = caminho(stem)
img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
fin = preds(p)
g = gt(stem, W, H)
print("exemplo:", [(nomes[c], round(s, 2)) for c, s, *_ in fin], "gt", len(g))
fig, ax = plt.subplots(figsize=(3.4, 3.4))
fig.subplots_adjust(0, 0, 1, 1)
ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
for c, x1, y1, x2, y2 in g:
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1.0, ls=(0, (3, 2))))
for c, s, x1, y1, x2, y2 in fin:
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=VERDE_DET, lw=1.4))
    ax.text(x1, y1 - 3, f"{PT[nomes[c]]} {fmt(s)}", fontsize=6.5, color="white", va="bottom", ha="left",
            bbox=dict(boxstyle="square,pad=0.15", fc=VERDE_DET, ec="none"))
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=VERDE_DET, lw=1.4, label="predição do modelo (classe e confiança)"),
                   Line2D([], [], color="white", lw=1.0, ls=(0, (3, 2)), label="anotação de referência")],
          loc="lower left", fontsize=6.0, framealpha=0.75, facecolor="#30343A", labelcolor="white", edgecolor="none")
salvar(fig, "exemplo_deteccao")
