#!/usr/bin/env python
"""Teste (D): bootstrap por IMAGEM do ΔAP por limiar de IoU (ASPE − base) e (E) pares de classes mais confundidos. VALIDAÇÃO."""
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, "scripts")
from diagnostico_aspe import load_gt, load_pred, iou_mat, ap101, THR, NC, TAU
NAMES = ["Ants","Bees","Beetles","Caterpillars","Earthworms","Earwigs","Grasshoppers","Moths","Slugs","Snails","Wasps","Weevils"]
gt = load_gt(); stems = list(gt)
def prep(name):
    P = load_pred(name); S = {c: [] for c in range(NC)}; T = {c: [] for c in range(NC)}; I = {c: [] for c in range(NC)}
    G = np.zeros((len(stems), NC)); conf = Counter()
    for ii, stem in enumerate(stems):
        g = gt[stem]; p = P.get(stem, np.zeros((0, 6)))
        for c in g[:, 0].astype(int): G[ii, c] += 1
        p = p[np.argsort(-p[:, 5])] if len(p) else p
        M = iou_mat(p[:, 1:5], g[:, 1:5]); same = (p[:, 0][:, None] == g[:, 0][None, :]) if len(p) and len(g) else np.zeros((len(p), len(g)), bool)
        Ms = np.where(same, M, 0.0); Tm = np.zeros((len(p), len(THR)), bool)
        for ti, t in enumerate(THR):
            used = set()
            for i in range(len(p)):
                cand = [(Ms[i, j], j) for j in range(len(g)) if j not in used and Ms[i, j] >= t]
                if cand: used.add(max(cand)[1]); Tm[i, ti] = True
        for i in range(len(p)):
            c = int(p[i, 0]); S[c].append(p[i, 5]); T[c].append(Tm[i]); I[c].append(ii)
        keep = p[:, 5] >= TAU if len(p) else np.zeros(0, bool)
        for j in range(len(g)):                      # confusões: GT sem acerto da própria classe e com caixa certa de outra classe
            if len(p[keep]) == 0: continue
            if Ms[keep][:, j].max() >= .5: continue
            o = M[keep][:, j] * (p[keep][:, 0] != g[j, 0]); k = o.argmax()
            if o[k] >= .5: conf[(int(g[j, 0]), int(p[keep][k, 0]))] += 1
    out = {}
    for c in range(NC):
        s = np.array(S[c]); o = np.argsort(-s); out[c] = (np.array(T[c])[o], np.array(I[c])[o])
    return out, G, conf
def boot_ap(d, G, m):
    AP = np.zeros((NC, len(THR))); ng = (m[:, None] * G).sum(0)
    for c in range(NC):
        T, I = d[c]; w = m[I].astype(float)
        for ti in range(len(THR)):
            ctp = np.cumsum(w * T[:, ti]); cfp = np.cumsum(w * ~T[:, ti])
            AP[c, ti] = ap101(ctp / ng[c], ctp / (ctp + cfp + 1e-9))
    return AP.mean(0)
rng = np.random.default_rng(1); N = len(stems)
for base, asp in (("plano_E01_s42", "plano_E02_s42"), ("plano_E03_s42", "plano_E04_s42")):
    db, Gb, cb = prep(base); da, Ga, ca = prep(asp)
    D = []
    for _ in range(300):
        m = np.bincount(rng.integers(0, N, N), minlength=N); D.append(boot_ap(da, Ga, m) - boot_ap(db, Gb, m))
    D = np.array(D); full = boot_ap(da, Ga, np.ones(N, int)) - boot_ap(db, Gb, np.ones(N, int))
    print(f"\n=== {asp} - {base}: bootstrap por imagem (300 reamostragens), ΔAP por IoU  [IC95%]")
    for ti, t in enumerate(THR):
        lo, hi = np.percentile(D[:, ti], [2.5, 97.5]); print(f"  IoU {t:.2f}: {full[ti]:+.4f} [{lo:+.4f},{hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
    dm = D.mean(1); lo, hi = np.percentile(dm, [2.5, 97.5]); print(f"  mAP50-95: {full.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  | mAP50: {full[0]:+.4f} [{np.percentile(D[:,0],2.5):+.4f},{np.percentile(D[:,0],97.5):+.4f}]")
    print("  pares (classe real → predita) mais alterados, nº de objetos:  base → ASPE")
    keys = set(cb) | set(ca)
    for k in sorted(keys, key=lambda k: -(ca[k] - cb[k]))[:6]: print(f"   {NAMES[k[0]]:>12} → {NAMES[k[1]]:<12} {cb[k]:>3} → {ca[k]:<3}")
    print(f"  total de confusões: {sum(cb.values())} → {sum(ca.values())}")
