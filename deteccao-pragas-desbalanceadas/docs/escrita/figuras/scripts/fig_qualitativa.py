"""Figuras da análise qualitativa E03 x E05 (protocolo: configs/qualitativa.yaml). SÓ VALIDAÇÃO, sementes 43/44.
Requer: python scripts/qualitativa.py preds && python scripts/qualitativa.py analise
Sorteios com numpy.random.default_rng(2026), na ordem do protocolo: painéis AgroPest-12 (a, depois b), raras do IP102,
mapas, galeria."""
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from matplotlib.patches import Rectangle

from estilo import *

ROOT = Path(RAIZ)
sys.path.insert(0, str(ROOT / "src"))
DET = pickle.load(open(ROOT / "runs" / "val_eval" / "qualitativa_det.pkl", "rb"))
RARAS = set(yaml.safe_load(open(ROOT / "configs" / "minoritarias_ip102_limpo.yaml", encoding="utf-8"))["faixas_de_frequencia"]["raras_menos_de_20"])
PT = ["formiga", "abelha", "besouro", "lagarta", "minhoca", "tesourinha", "gafanhoto", "mariposa", "lesma", "caracol", "vespa", "gorgulho"]
IPN = yaml.safe_load(open(ROOT / "data" / "processed" / "ip102_limpo" / "data.yaml", encoding="utf-8"))["names"]
COR_OK, COR_CLS, COR_FP = "#009E73", "#D55E00", "#0072B2"   # Okabe-Ito; validados (scripts/validate_palette.js)
rng = np.random.default_rng(2026)


def salvar_foto(fig, nome_fig):
    """Figuras com fotografias: JPG (arquivo leve para o Overleaf)."""
    import os
    caminho = os.path.join(OUT, f"{nome_fig}.jpg")
    fig.savefig(caminho, dpi=250, bbox_inches="tight", pad_inches=0.03, facecolor="white", pil_kwargs={"quality": 88})
    plt.close(fig)
    print("salvo:", caminho, f"{os.path.getsize(caminho) / 1e6:.2f} MB")


def nome(base, c):
    if base == "agropest12":
        return PT[c]
    n = IPN[c]
    return f"{c}: {n if len(n) <= 18 else n[:17] + '…'}"


def fmt(x):
    return f"{x:.2f}".replace(".", ",")


def categorias(k):
    """{stem: [(classe, cat_E03, cat_E05, iou3, iou5)]} para a base/semente k."""
    d = {}
    for x3, x5 in zip(DET[k]["o3"], DET[k]["o5"]):
        d.setdefault(x3[0], []).append((x3[2], x3[3], x5[3], x3[4], x5[4]))
    return d


def iou1(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def painel(ax, base, k, stem, quem, titulo=None):
    g, path = DET[k]["gt"][stem]
    im = Image.open(path).convert("RGB")
    W, H = im.size
    ax.imshow(im)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_color(CINZA)
    fs = 5.2
    rotula = len(g) <= 6          # imagens com muitos objetos ficam sem texto, para não encobrir os insetos
    ocupados = []

    def texto(x, y, s, cor_fundo, cor_txt):
        passo = 0.055 * max(W, H)   # altura aproximada de uma linha de rótulo, em pixels da imagem
        for (ox, oy) in ocupados:  # evita sobrepor rótulos de caixas vizinhas
            if abs(ox - x) < 0.45 * W and abs(oy - y) < passo:
                y = oy + passo
        ocupados.append((x, y))
        ax.text(x, y, s, fontsize=fs, va="top", ha="left", color=cor_txt, clip_on=True,
                bbox=dict(fc=cor_fundo, ec="none", pad=0.5, alpha=0.88))

    if quem == "ref":
        varias = len({int(c) for c in g[:, 0]}) > 1 or base == "ip102"
        for c, x1, y1, x2, y2 in g:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="black", lw=2.0))
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1.0))
            if rotula and varias:
                texto(x1 + 0.01 * W, y1 + 0.01 * H, nome(base, int(c)), "white", "black")
    else:
        cat = categorias(k)[stem]
        p, st = DET[k]["s3" if quem == "E03" else "s5"][stem]
        idx = 1 if quem == "E03" else 2
        for j, (c, x1, y1, x2, y2) in enumerate(g):   # objetos não acertados: contorno pontilhado
            if cat[j][idx] != "ok":
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1.1, ls=(0, (2, 2))))
        for (c, x1, y1, x2, y2, s), stt in zip(p, st):
            cor = {"ok": COR_OK, "cls": COR_CLS, "fp": COR_FP}[stt]
            ls = "-" if stt != "fp" else (0, (4, 2))
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=cor, lw=1.5, ls=ls))
            if not rotula:
                continue
            if stt == "ok":
                best = max(iou1((x1, y1, x2, y2), gg[1:]) for gg in g if int(gg[0]) == int(c))
                txt = f"{fmt(s)} | IoU {fmt(best)}"
            else:
                txt = f"{nome(base, int(c))} {fmt(s)}"
            texto(max(x1, 0) + 0.01 * W, max(y1, 0) + 0.01 * H, txt, cor, "white")
    if titulo:
        ax.set_title(titulo, fontsize=7, pad=3)


def grade(base, k, stems, nome_fig, rotulos=None):
    n = len(stems)
    ars = []
    for s in stems:
        w_, h_ = Image.open(DET[k]["gt"][s][1]).size
        ars.append(h_ / w_)
    larg, leg, tit = 6.3, 0.32, 0.22
    alt_p = (larg - 0.25) / 3                     # largura útil de cada painel (pol.)
    Hfig = sum(alt_p * a for a in ars) * 1.04 + leg + tit
    fig = plt.figure(figsize=(larg, Hfig))
    gs = fig.add_gridspec(n, 3, height_ratios=ars, wspace=0.03, hspace=0.04,
                          left=0.25 / larg, right=0.999, top=1 - tit / Hfig, bottom=leg / Hfig)
    for i, s in enumerate(stems):
        for j, quem in enumerate(("ref", "E03", "E05")):
            ax = fig.add_subplot(gs[i, j])
            tit_ = ("Referência", "E03 (YOLO26s)", "E05 (ASPE residual)")[j] if i == 0 else None
            painel(ax, base, k, s, quem, tit_)
            if j == 0 and rotulos:
                ax.set_ylabel(rotulos[i], fontsize=6.3, labelpad=3)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COR_OK, lw=1.5, label="acerto (classe certa, IoU ≥ 0,5)"),
         Line2D([], [], color=COR_CLS, lw=1.5, label="classe trocada"),
         Line2D([], [], color=COR_FP, lw=1.5, ls=(0, (4, 2)), label="falso positivo"),
         Line2D([], [], color=CINZA, lw=1.1, ls=(0, (2, 2)), label="objeto não acertado")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    salvar_foto(fig, nome_fig)


# ---------------------------------------------------------------- 1. AgroPest-12: categorias (a) e (b), semente 43
k = "agropest12_s43"
C = categorias(k)
usados_cls, escolhidos, rot = set(), [], []
for alvo, (i3ok, i5ok) in (("a", (False, True)), ("b", (True, False))):
    cand = [(s, c) for s, objs in C.items() for (c, c3, c5, _, _) in objs if (c3 == "ok") == i3ok and (c5 == "ok") == i5ok]
    cand = [cand[i] for i in rng.permutation(len(cand))]
    n = 0
    for s, c in cand:
        if c in usados_cls or s in escolhidos:
            continue
        usados_cls.add(c); escolhidos.append(s); rot.append(f"({alvo}) {PT[c]}"); n += 1
        if n == 2:
            break
grade("agropest12", k, escolhidos, "qual_agropest_paineis", rot)
print("AgroPest-12 painéis:", list(zip(escolhidos, rot)))

# ---------------------------------------------------------------- 2. IP102: faixa rara, semente 43
k2 = "ip102_s43"
C2 = categorias(k2)
cand = sorted(s for s, objs in C2.items() if any(o[0] in RARAS for o in objs))
rar = [cand[i] for i in rng.choice(len(cand), 4, replace=False)]
def rot_raras(s):
    o = [x for x in C2[s] if x[0] in RARAS]
    k3 = sum(x[1] == "ok" for x in o); k5 = sum(x[2] == "ok" for x in o)
    return f"acertos: E03 {k3}/{len(o)}, E05 {k5}/{len(o)}"


rot2 = [rot_raras(s) for s in rar]
grade("ip102", k2, rar, "qual_ip102_raras", rot2)
print("IP102 raras (candidatas", len(cand), "):", list(zip(rar, rot2)))

# ---------------------------------------------------------------- 3. diferença de IoU nos objetos (c), small multiples
fig, axs = plt.subplots(2, 2, figsize=(6.3, 3.6), sharey="row", gridspec_kw=dict(hspace=0.55, wspace=0.08))
for i, base in enumerate(("agropest12", "ip102")):
    for j, sem in enumerate((43, 44)):
        kk = f"{base}_s{sem}"
        d = np.array([x5[4] - x3[4] for x3, x5 in zip(DET[kk]["o3"], DET[kk]["o5"]) if x3[3] == "ok" and x5[3] == "ok"])
        ax = axs[i, j]
        ax.hist(np.clip(d, -0.3, 0.3), bins=np.linspace(-0.3, 0.3, 41), color="#8A9099", ec="white", lw=0.4)
        ax.axvline(0, color=TINTA, lw=0.8)
        med = np.median(d)
        ax.axvline(med, color=VERMELHO, lw=1.2, ls=(0, (3, 1.5)), label="mediana")
        ax.set_title(f"{'AgroPest-12' if base == 'agropest12' else 'IP102'}, semente {sem}", fontsize=7.2, pad=3)
        txt = f"n = {len(d)}\nmediana = {med:+.3f}\nIoU maior no E05: {100 * (d > 0).mean():.0f}%"
        ax.text(0.98, 0.95, txt.replace(".", ",").replace("-", "−"),
                transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=TINTA)
        if (i, j) == (0, 0):
            ax.legend(loc="upper left", fontsize=6.2, frameon=False, handlelength=1.8)
        ax.tick_params(labelsize=6.3, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i == 1:
            ax.set_xlabel("IoU do E05 − IoU do E03 (mesmo objeto)", fontsize=6.8)
        if j == 0:
            ax.set_ylabel("Objetos", fontsize=6.8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",").replace("-", "−")))
salvar(fig, "qual_iou_diferenca")

# ---------------------------------------------------------------- 4. mapas: EigenCAM (E03, E05) e ramo ASPE do E05
import torch
from tcc.models.yolo_aspe import load_fp32
torch.set_num_threads(8)
info = lambda r: yaml.safe_load(open(ROOT / "runs" / "logs" / f"plano_{r}.yaml", encoding="utf-8"))
m3 = load_fp32(Path(info("plano_E03_s43")["best"]).with_name("best_fp32.pt"))[0].model
m5 = load_fp32(Path(info("plano_E04_s43_ares")["best"]).with_name("best_fp32.pt"))[0].model
assert type(m3.model[10]).__name__ == type(m5.model[10]).__name__, "camada 10 diferente"
print("camada 10:", type(m3.model[10]).__name__, "| camada 6 do E05:", type(m5.model[6]).__name__)
cand = sorted(s for s, objs in C.items() if all(o[1] == "ok" and o[2] == "ok" for o in objs))
cand = [s for s in cand if s not in escolhidos]
mp = [cand[i] for i in rng.choice(len(cand), 3, replace=False)]


def eigencam(a):
    a = a[0].detach().numpy()                       # (C, H, W)
    Cc, Hh, Ww = a.shape
    X = a.reshape(Cc, -1).T                        # (HW, C)
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    p = (Xc @ vt[0]).reshape(Hh, Ww)
    if np.corrcoef(p.ravel(), a.mean(0).ravel())[0, 1] < 0:   # o sinal da componente é arbitrário; fixa-se pela ativação média
        p = -p
    p = np.maximum(p, 0)
    return p / (p.max() + 1e-9)


calc = []
for s in mp:
    g, path = DET[k]["gt"][s]
    im = Image.open(path).convert("RGB")
    x = torch.from_numpy(np.asarray(im.resize((640, 640)), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    acts = {}
    h1 = m3.model[10].register_forward_hook(lambda mod, inp, out: acts.__setitem__("e3", out))
    h2 = m5.model[10].register_forward_hook(lambda mod, inp, out: acts.__setitem__("e5", out))
    h3 = m5.model[6].aspe.register_forward_hook(lambda mod, inp, out: acts.__setitem__("ramo", (inp[0], out)))
    with torch.no_grad():
        m3(x); m5(x)
    for hh in (h1, h2, h3):
        hh.remove()
    y_in, r = acts["ramo"]
    rel = (r[0].norm(dim=0) / (y_in[0].norm(dim=0) + 1e-6)).numpy()   # |ASPE(y)| / |y| por posição
    calc.append((g, im, [None, eigencam(acts["e3"]), eigencam(acts["e5"]), rel]))
esc_ramo = max(c[2][3].max() for c in calc)   # mesma escala nas três imagens
print("ramo ASPE: razão |ASPE(y)|/|y| máxima por imagem:", [round(float(c[2][3].max()), 3) for c in calc],
      "| mediana:", [round(float(np.median(c[2][3])), 3) for c in calc])
fig, axs = plt.subplots(3, 4, figsize=(6.3, 5.25), gridspec_kw=dict(wspace=0.03, hspace=0.06, bottom=0.1))
vmax_ramo = np.ceil(esc_ramo * 10) / 10
for i, (g, im, mapas) in enumerate(calc):
    for j in range(4):
        ax = axs[i, j]
        if j < 3:
            ax.imshow(im)
        if j in (1, 2):   # EigenCAM sobre a imagem, normalizado em cada imagem
            mm = np.asarray(Image.fromarray((255 * np.clip(mapas[j], 0, 1)).astype(np.uint8)).resize(im.size, Image.BILINEAR)) / 255
            ax.imshow(mm, cmap="magma", alpha=0.55, vmin=0, vmax=1)
        if j == 3:        # razão |ASPE(y)|/|y| sozinha, na mesma escala absoluta nas três imagens
            ramo = ax.imshow(mapas[3], cmap="magma", vmin=0, vmax=vmax_ramo, extent=(0, im.size[0], im.size[1], 0),
                             interpolation="bilinear")
        for c, x1, y1, x2, y2 in g:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=0.9))
        ax.set_xlim(0, im.size[0]); ax.set_ylim(im.size[1], 0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color(CINZA)
        if i == 0:
            ax.set_title(("Imagem e referência", "EigenCAM, E03", "EigenCAM, E05", "Ramo ASPE do E05")[j], fontsize=7, pad=3)
cax = fig.add_axes([axs[2, 3].get_position().x0, 0.055, axs[2, 3].get_position().width, 0.014])
cb = fig.colorbar(ramo, cax=cax, orientation="horizontal")
cb.ax.tick_params(labelsize=5.8, length=2)
cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
cb.set_label("|ASPE(y)| / |y|", fontsize=6.2, labelpad=1)
salvar_foto(fig, "qual_mapas")
print("mapas:", mp)

# ---------------------------------------------------------------- 5. galeria (apêndice)
excl = set(escolhidos) | set(mp)
gal = []
for c in range(12):
    cand = sorted(s for s, objs in C.items() if any(o[0] == c for o in objs) and s not in excl and s not in gal)
    gal.append(cand[rng.integers(len(cand))])
grade("agropest12", k, gal[:6], "qual_galeria_agropest_1", [PT[c] for c in range(6)])
grade("agropest12", k, gal[6:], "qual_galeria_agropest_2", [PT[c] for c in range(6, 12)])
cand = sorted(s for s, objs in C2.items() if any(o[0] in RARAS for o in objs) and s not in rar)
gi = [cand[i] for i in rng.choice(len(cand), 6, replace=False)]
rot3 = [rot_raras(s) for s in gi]
grade("ip102", k2, gi, "qual_galeria_ip102", rot3)
print("galeria:", gal, gi)
