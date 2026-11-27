"""Figuras do Capítulo 4 (resultados). Usa apenas resultados já registrados em runs/ (validação e o teste único de 18/11/2026)."""
import csv
import json
import os
import numpy as np
from estilo import *

ROOT = RAIZ
from matplotlib.ticker import FuncFormatter
def virgula(ax, nd=2):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.{nd}f}".replace(".", ",").replace("-", "−")))


COR = {"E01": "#8A9099", "E02": "#C27BD9", "E03": AZUL[1], "E04": "#D9822B", "E05": ROXO[1]}


def curva(run):
    import yaml as _y
    pasta = _y.safe_load(open(os.path.join(ROOT, "runs", "logs", f"plano_{run}.yaml"), encoding="utf-8"))["save_dir"]  # pasta oficial do treino
    rows = list(csv.DictReader(open(os.path.join(pasta, "results.csv"), encoding="utf-8")))
    k50 = next(k for k in rows[0] if "mAP50(B)" in k)
    k95 = next(k for k in rows[0] if "mAP50-95(B)" in k)
    ep = np.array([int(float(r["epoch"])) for r in rows])
    f = np.array([0.1 * float(r[k50]) + 0.9 * float(r[k95]) for r in rows])
    return ep, f


# ---------------- 1. curvas de treino E03 / E04 / E05 ----------------
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.5))
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.19, top=0.86, wspace=0.25)
for ax, (titulo, pref) in zip(axs, [("(a) AgroPest-12", "plano"), ("(b) IP102", "ip102_limpo")]):
    for nome, run in [("E03 (YOLO26s)", f"{pref}_E03_s42"), ("E04 (ASPE original)", f"{pref}_E04_s42"),
                      ("E05 (ASPE residual)", f"{pref}_E04_s42_ares")]:
        ep, f = curva(run)
        ax.plot(ep, f, lw=1.1, color=COR[nome[:3]], label=nome)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Época", fontsize=7)
    ax.set_title(titulo, fontsize=8, weight="bold", pad=4)
    ax.tick_params(labelsize=6.5, length=2.5)
    ax.yaxis.grid(True, color="#E3E6EA", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel(r"$\mathit{Fitness}$ na validação", fontsize=7)
for _a in axs: virgula(_a)
axs[1].legend(fontsize=6.3, frameon=False, loc="lower right")
salvar(fig, "curvas_treino_aspe")

# ---------------- 2. ΔAP por limiar de IoU (AgroPest-12, validação) ----------------
a = json.load(open(os.path.join(ROOT, "runs", "val_eval", "ap_por_iou.json"), encoding="utf-8"))
iou = np.arange(0.50, 0.96, 0.05)
pares = [("E02 − E01 (ASPE original no YOLOv8s)", "plano_E02_s42", "plano_E01_s42", COR["E02"], "o"),
         ("E04 − E03 (ASPE original no YOLO26s)", "plano_E04_s42", "plano_E03_s42", COR["E04"], "s"),
         ("E05 − E03 (ASPE residual no YOLO26s)", "plano_E04_s42_ares", "plano_E03_s42", COR["E05"], "D")]
fig, ax = plt.subplots(figsize=(5.2, 2.6))
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.19, top=0.97)
ax.axhline(0, color=TINTA, lw=0.7)
for nome, x, y, cor, mk in pares:
    d = np.array(a[x]) - np.array(a[y])
    ax.plot(iou, d, color=cor, lw=1.2, marker=mk, ms=3.2, label=nome)
ax.set_xticks(iou); ax.set_xticklabels([f"{v:.2f}".replace(".", ",") for v in iou])
ax.set_xlabel("Limiar de IoU", fontsize=7)
ax.set_ylabel("Diferença de AP", fontsize=7)
ax.tick_params(labelsize=6.3, length=2.5)
ax.yaxis.grid(True, color="#E3E6EA", lw=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.2, frameon=False, loc="upper left")
virgula(ax)
salvar(fig, "delta_ap_por_iou")

# ---------------- 3. E03 × E05 por semente, validação e teste ----------------
v = json.load(open(os.path.join(ROOT, "runs", "val_eval", "plano_val.json"), encoding="utf-8"))
t = json.load(open(os.path.join(ROOT, "runs", "val_eval", "teste_final.json"), encoding="utf-8"))
fit = lambda m: 0.1 * m["map50"] + 0.9 * m["map50_95"]
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.55))
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.86, wspace=0.28)
for ax, (titulo, pref) in zip(axs, [("(a) AgroPest-12", "plano"), ("(b) IP102", "ip102_limpo")]):
    xs, lab = [], []
    for g, (fonte, nomefonte) in enumerate([(v, "validação"), (t, "teste")]):
        for i, s in enumerate((42, 43, 44)):
            x = g * 3.8 + i
            e3 = fit(fonte[f"{pref}_E03_s{s}"]); e5 = fit(fonte[f"{pref}_E04_s{s}_ares"])
            ax.plot([x, x], [e3, e5], color="#B9BEC5", lw=1.0, zorder=1)
            ax.plot(x, e3, "o", color=COR["E03"], ms=4.2, zorder=2, label="E03 (YOLO26s)" if (g, i) == (0, 0) else None)
            ax.plot(x, e5, "D", color=COR["E05"], ms=3.8, zorder=3, label="E05 (ASPE residual)" if (g, i) == (0, 0) else None)
            xs.append(x); lab.append(str(s))
        ax.text(g * 3.8 + 1, 1.0, nomefonte, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.6,
                color=CINZA)
    ax.set_xticks(xs); ax.set_xticklabels(lab)
    ax.set_xlabel("Semente", fontsize=7)
    ax.set_title(titulo, fontsize=8, weight="bold", pad=12)
    ax.tick_params(labelsize=6.3, length=2.5)
    ax.yaxis.grid(True, color="#E3E6EA", lw=0.6)
    ax.set_xlim(-0.6, 6.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel(r"$\mathit{Fitness}$", fontsize=7)
h, l = axs[0].get_legend_handles_labels()
axs[1].legend(h, l, fontsize=6.2, frameon=False, loc="upper right")
for _a in axs: virgula(_a, 3)
salvar(fig, "sementes_e03_e05")

# ---------------- 4. AP por faixa no teste do IP102 ----------------
import yaml
C = yaml.safe_load(open(os.path.join(ROOT, "configs", "minoritarias_ip102_limpo.yaml"), encoding="utf-8"))
NAMES = yaml.safe_load(open(os.path.join(ROOT, "data", "processed", "ip102_limpo", "data.yaml"), encoding="utf-8"))["names"]
faixas = list(C["faixas_de_frequencia"].items())


def bandas(m):
    pc = m["por_classe"]
    return [np.mean([pc[NAMES[c]]["ap50_95"] for c in cl if NAMES[c] in pc]) for _, cl in faixas]


grupos = [("E03\n(3 sementes)", [f"ip102_limpo_E03_s{s}" for s in (42, 43, 44)], COR["E03"]),
          ("E05\n(3 sementes)", [f"ip102_limpo_E04_s{s}_ares" for s in (42, 43, 44)], COR["E05"]),
          ("E05 +\nMixUp 0,3", ["ip102_limpo_E04_s42_ares_MIX"], "#2E9E6A"),
          ("E05 +\nRFS", ["ip102_limpo_E04_s42_ares_RFS"], "#D9822B"),
          ("E05 +\nCopy-Paste", ["ip102_limpo_E04_s42_ares_CP"], "#C0392B"),
          ("E05 +\nRFS+CP", ["ip102_limpo_E04_s42_ares_RFS_CP"], "#8E5A2B")]
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.45), sharey=False)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.88, wspace=0.34)
for j, (ax, (nome_faixa, _)) in enumerate(zip(axs, faixas)):
    for i, (rot, runs, cor) in enumerate(grupos):
        vals = np.array([bandas(t[r])[j] for r in runs])
        ax.bar(i, vals.mean(), color=cor, width=0.7, zorder=2, alpha=0.9)
        if len(vals) > 1:
            ax.errorbar(i, vals.mean(), yerr=[[vals.mean() - vals.min()], [vals.max() - vals.mean()]], color=TINTA,
                        lw=0.8, capsize=2, zorder=3)
    titulo = {"raras_menos_de_20": "Raras (< 20 imagens)", "comuns_20_a_99": "Comuns (20 a 99)",
              "frequentes_100_ou_mais": "Frequentes (≥ 100)"}[nome_faixa]
    ax.set_title(titulo, fontsize=7.6, weight="bold", pad=3)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=6.2, length=2.5); ax.tick_params(axis="x", length=0)
    lo = min(bandas(t[r])[j] for _, runs, _ in grupos for r in runs)
    ax.set_ylim(max(0, lo - 0.05), None)
    ax.yaxis.grid(True, color="#E3E6EA", lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("AP@0,5:0,95 no teste", fontsize=7)
for _a in axs: virgula(_a)
from matplotlib.patches import Patch
fig.legend([Patch(color=g[2]) for g in grupos], [" ".join(g[0].split()) for g in grupos], loc="lower center", ncol=6,
           fontsize=6.0, frameon=False, bbox_to_anchor=(0.53, 0.0), handlelength=1.2, columnspacing=1.1)
salvar(fig, "faixas_ip102_teste")
