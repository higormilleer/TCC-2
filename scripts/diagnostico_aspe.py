#!/usr/bin/env python
"""Por que o ASPE perde mAP50 e (no AgroPest-12) ganha em IoU alto? Análise sobre as predições da VALIDAÇÃO.
Requer: python scripts/dump_preds_val.py <runs>.  Uso: python scripts/diagnostico_aspe.py plano_E01_s42 plano_E02_s42 ..."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
TAU = 0.25
THR = np.arange(0.5, 0.96, 0.05)
DS = ROOT / "data" / "processed" / "agropest12_plano"
NC = 12


def load_gt():
    gt = {}
    for lf in sorted((DS / "labels" / "val").glob("*.txt")):
        im = next(iter((DS / "images" / "val").glob(lf.stem + ".*")))
        W, H = Image.open(im).size
        rows = []
        for l in lf.read_text().splitlines():
            p = l.split()
            if len(p) >= 5:
                c, x, y, w, h = int(float(p[0])), *map(float, p[1:5])
                rows.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H, w * h))
        gt[lf.stem] = np.array(rows, dtype=float).reshape(-1, 6)
    return gt


def load_pred(name):
    J = json.loads((ROOT / "runs" / "val_eval" / "pred" / name / "predictions.json").read_text())
    off = 0 if min(j["category_id"] for j in J) == 0 else 1
    P = {}
    for j in J:
        x, y, w, h = j["bbox"]
        P.setdefault(str(j["image_id"]), []).append((j["category_id"] - off, x, y, x + w, y + h, j["score"]))
    return {k: np.array(v) for k, v in P.items()}


def iou_mat(a, b):  # a: (n,4) b: (m,4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def ap101(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return np.trapz(np.interp(x, mrec, mpre), x)


def analyse(name, gt):
    P = load_pred(name)
    # --- AP por classe e limiar (casamento guloso por score, mesma classe)
    sc = {c: [] for c in range(NC)}; tp = {c: [] for c in range(NC)}; ngt = np.zeros(NC)
    per_img = []
    for stem, g in gt.items():
        p = P.get(stem, np.zeros((0, 6)))
        for c in g[:, 0].astype(int): ngt[c] += 1
        order = np.argsort(-p[:, 5]) if len(p) else []
        p = p[order] if len(p) else p
        M = iou_mat(p[:, 1:5], g[:, 1:5])
        same = (p[:, 0][:, None] == g[:, 0][None, :]) if len(p) and len(g) else np.zeros((len(p), len(g)), bool)
        Ms = np.where(same, M, 0.0)
        T = np.zeros((len(p), len(THR)), bool)
        for ti, t in enumerate(THR):
            used = set()
            for i in range(len(p)):
                cand = [(Ms[i, j], j) for j in range(len(g)) if j not in used and Ms[i, j] >= t]
                if cand:
                    j = max(cand)[1]; used.add(j); T[i, ti] = True
        for i in range(len(p)):
            sc[int(p[i, 0])].append(p[i, 5]); tp[int(p[i, 0])].append(T[i])
        # --- operação em conf >= TAU
        keep = p[:, 5] >= TAU if len(p) else np.zeros(0, bool)
        pk, Mk, Msk = p[keep], M[keep] if len(p) else M, Ms[keep] if len(p) else Ms
        rec = {"n": len(g), "bsame": np.zeros(len(g)), "cat": [""] * len(g), "area": g[:, 5]}
        for j in range(len(g)):
            bs = Msk[:, j].max() if len(pk) else 0.0
            other = (Mk[:, j] * (pk[:, 0] != g[j, 0])).max() if len(pk) else 0.0
            rec["bsame"][j] = bs
            rec["cat"][j] = "TP" if bs >= .5 else ("cls" if other >= .5 else ("loc" if bs >= .1 else "miss"))
        # falsos positivos em TAU (casamento IoU .5)
        fp = {"bg": 0, "loc": 0, "cls": 0, "dup": 0}; matched = set()
        for i in range(len(pk)):
            c = pk[i, 0]; cand = [(Msk[i, j], j) for j in range(len(g)) if j not in matched and Msk[i, j] >= .5]
            if cand:
                matched.add(max(cand)[1]); continue
            if len(g) == 0 or Mk[i].max() < .1: fp["bg"] += 1
            elif Msk[i].max() >= .5: fp["dup"] += 1
            elif (Mk[i] * (g[:, 0] != c)).max() >= .5: fp["cls"] += 1
            else: fp["loc"] += 1
        rec["fp"] = fp; rec["npred"] = len(pk)
        per_img.append(rec)
    AP = np.zeros((NC, len(THR)))
    for c in range(NC):
        if ngt[c] == 0 or not sc[c]: continue
        s = np.array(sc[c]); T = np.array(tp[c]); o = np.argsort(-s); T = T[o]
        for ti in range(len(THR)):
            ctp = np.cumsum(T[:, ti]); cfp = np.cumsum(~T[:, ti])
            AP[c, ti] = ap101(ctp / ngt[c], ctp / (ctp + cfp + 1e-9))
    return AP, per_img


def summarize(per_img):
    n = sum(r["n"] for r in per_img)
    b = np.concatenate([r["bsame"] for r in per_img])
    cat = sum((r["cat"] for r in per_img), [])
    area = np.concatenate([r["area"] for r in per_img])
    fp = {k: sum(r["fp"][k] for r in per_img) for k in ("bg", "loc", "cls", "dup")}
    d = {"n_gt": n, "R50": (b >= .5).mean(), "R75": (b >= .75).mean(), "R85": (b >= .85).mean(), "R90": (b >= .9).mean(),
         "IoU_medio_TP": b[b >= .5].mean(), "n_pred": sum(r["npred"] for r in per_img),
         **{"GT_" + k: cat.count(k) / n for k in ("TP", "cls", "loc", "miss")}, **{"FP_" + k: v for k, v in fp.items()}}
    return d, b, np.array(cat), area


if __name__ == "__main__":
    names = sys.argv[1:]
    gt = load_gt(); res = {}
    for nm in names:
        AP, pi = analyse(nm, gt)
        d, b, cat, area = summarize(pi)
        res[nm] = (AP, pi, d, b, cat, area)
        print(f"\n== {nm} | meu mAP50 {AP[:, 0].mean():.4f} mAP50-95 {AP.mean():.4f} (compare com o val do Ultralytics)")
        print({k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in d.items()})
    import pickle
    pickle.dump({k: (v[0], v[2], v[3], v[4], v[5]) for k, v in res.items()}, open(ROOT / "runs" / "val_eval" / "diag_aspe.pkl", "wb"))
