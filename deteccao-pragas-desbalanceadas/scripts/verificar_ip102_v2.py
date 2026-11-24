#!/usr/bin/env python
"""Verifica o IP102 regenerado (data/processed/ip102_v2): mapeamento de classes, distribuição e vazamento entre splits.

1) Mapeamento: o id de classe de cada caixa nos rótulos YOLO deve ser IGUAL ao <name> (0-based) do XML original.
2) Distribuição de instâncias por classe em treino/val/teste e razão de desbalanceamento.
3) Vazamento: imagens idênticas ou quase idênticas (dHash, distância de Hamming <= 3) entre splits diferentes.
Uso: python scripts/verificar_ip102_v2.py
Grava docs/auditoria/ip102_v2_verificacao.json
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ip102" / "VOC2007"
OUT = ROOT / "data" / "processed" / "ip102_v2"

# ---- 1) mapeamento ----
xml_cls = {}
for x in sorted((RAW / "Annotations").glob("*.xml")):
    try:
        r = ET.parse(x).getroot()
    except ET.ParseError:
        continue
    stem = Path((r.findtext("filename") or x.stem).strip()).stem
    xml_cls[stem] = sorted(int(o.findtext("name")) for o in r.findall("object") if (o.findtext("name") or "").strip().isdigit())

ok = bad = 0
exemplos = []
lab_dir = OUT / "labels_all"
for f in lab_dir.glob("*.txt"):
    ids = sorted(int(float(l.split()[0])) for l in f.read_text().splitlines() if l.strip())
    esperado = xml_cls.get(f.stem)
    if esperado is None:
        continue
    # caixas inválidas podem ter sido descartadas: o rótulo deve ser sub-multiconjunto do XML
    if not (Counter(ids) - Counter(esperado)) and ids:
        ok += 1
    else:
        bad += 1
        if len(exemplos) < 5:
            exemplos.append((f.stem, ids[:5], esperado[:5]))
print(f"mapeamento: {ok} imagens consistentes com o XML, {bad} inconsistentes {exemplos}")

# ---- 2) distribuição ----
manifest = json.loads((ROOT / "data" / "splits" / "ip102_v2" / "manifest.json").read_text(encoding="utf-8"))
dist = {}
for sp in ("train", "val", "holdout"):
    c = Counter()
    for stem in manifest[sp]:
        for l in (lab_dir / f"{stem}.txt").read_text().splitlines():
            if l.strip():
                c[int(float(l.split()[0]))] += 1
    dist[sp] = c
    v = [c.get(i, 0) for i in range(102)]
    nz = [x for x in v if x > 0]
    print(f"{sp}: {len(manifest[sp])} imagens, {sum(v)} instâncias, classes com instâncias: {len(nz)}/102, "
          f"min {min(nz)} max {max(nz)} razão {max(nz) / min(nz):.0f}:1, classes sem nenhuma: {[i for i in range(102) if v[i] == 0]}")

# ---- 3) vazamento (dHash) ----
def dhash(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    s = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


imgs = {sp: [] for sp in ("train", "val", "holdout")}
hashes = {}
for sp in imgs:
    for stem in manifest[sp]:
        p = next(iter((OUT / "images_all").glob(stem + ".*")), None)
        h = dhash(p) if p else None
        if h is not None:
            imgs[sp].append(stem); hashes[stem] = h
arr = {sp: np.array([hashes[s] for s in imgs[sp]], dtype=np.uint64) for sp in imgs}


def popcount(a):
    a = a - ((a >> np.uint64(1)) & np.uint64(0x5555555555555555))
    a = (a & np.uint64(0x3333333333333333)) + ((a >> np.uint64(2)) & np.uint64(0x3333333333333333))
    a = (a + (a >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((a * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(int)


def cruzado(a, b, thr=3):
    n_a = 0
    for i in range(0, len(a), 500):
        d = popcount(a[i:i + 500, None] ^ b[None, :])
        n_a += int((d.min(axis=1) <= thr).sum())
    return n_a


leak = {}
for a, b in (("val", "train"), ("holdout", "train"), ("holdout", "val")):
    leak[f"{a}->{b}"] = {"identicas_ou_quase(<=3 bits)": cruzado(arr[a], arr[b]), "n_{}".format(a): len(arr[a])}
    print(f"vazamento {a} -> {b}: {leak[f'{a}->{b}']}")

res = {"mapeamento_ok": ok, "mapeamento_inconsistente": bad, "exemplos_inconsistentes": exemplos,
       "n_imagens": {k: len(v) for k, v in manifest.items() if k in ("train", "val", "holdout")},
       "distribuicao_instancias": {sp: {str(k): v for k, v in sorted(c.items())} for sp, c in dist.items()}, "vazamento": leak}
(ROOT / "docs" / "auditoria" / "ip102_v2_verificacao.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
