#!/usr/bin/env python
"""Gera data/processed/ip102_limpo: IP102 sem imagens idênticas/quase idênticas (dHash, Hamming <= 3) entre splits.

Regras (o TESTE OFICIAL do IP102 fica intacto):
  1. agrupa por union-find todas as imagens com distância de Hamming <= 3;
  2. grupo que contém imagem do teste: o teste fica; os membros de treino/val do grupo são REMOVIDOS;
  3. grupo só com treino e val: os membros de val passam para o treino (nenhuma imagem é descartada nesse caso).
Uso: python scripts/limpar_vazamento_ip102.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcc.data.prepare import _materialize_yolo_layout  # noqa: E402
from tcc.data.yolo_io import write_ultralytics_yaml  # noqa: E402

SRC = ROOT / "data" / "processed" / "ip102_v2"
DST = ROOT / "data" / "processed" / "ip102_limpo"
man = json.loads((ROOT / "data" / "splits" / "ip102_v2" / "manifest.json").read_text(encoding="utf-8"))
split_of = {}
for sp in ("train", "val", "holdout"):
    for s in man[sp]:
        split_of[s] = sp
stems = sorted(split_of)


def dhash(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    s = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    return int(np.packbits((s[:, 1:] > s[:, :-1]).flatten()).view(">u8")[0])


hs, ok = [], []
for s in stems:
    p = next(iter((SRC / "images_all").glob(s + ".*")), None)
    h = dhash(p) if p else None
    if h is not None:
        hs.append(h); ok.append(s)
stems = ok
H = np.array(hs, dtype=np.uint64)
print("imagens com hash:", len(stems))


def popcount(a):
    a = a - ((a >> np.uint64(1)) & np.uint64(0x5555555555555555))
    a = (a & np.uint64(0x3333333333333333)) + ((a >> np.uint64(2)) & np.uint64(0x3333333333333333))
    a = (a + (a >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((a * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(int)


parent = list(range(len(stems)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


npairs = 0
for i in range(0, len(H), 400):
    d = popcount(H[i:i + 400, None] ^ H[None, :])
    ii, jj = np.nonzero(d <= 3)
    for a, b in zip(ii + i, jj):
        if a < b:
            npairs += 1
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
print("pares com Hamming <= 3:", npairs)

grupos = {}
for i, s in enumerate(stems):
    grupos.setdefault(find(i), []).append(s)
tam = sorted((len(g) for g in grupos.values()), reverse=True)
print("grupos:", len(grupos), "| maiores:", tam[:8], "| grupos com > 1 imagem:", sum(1 for t in tam if t > 1))

train, val, test = [], [], []
removidos = {"train": 0, "val": 0}
for g in grupos.values():
    sps = {split_of[s] for s in g}
    if "holdout" in sps:
        for s in g:
            if split_of[s] == "holdout":
                test.append(s)
            else:
                removidos[split_of[s]] += 1
    elif "val" in sps and "train" in sps:
        train.extend(g)
    else:
        (train if "train" in sps else val).extend(g)

print(f"treino {len(train)} | val {len(val)} | teste {len(test)} | removidos do treino {removidos['train']}, da val {removidos['val']}")
names = man["names"]
layout = _materialize_yolo_layout(DST, SRC / "images_all", SRC / "labels_all", sorted(train), sorted(val), sorted(test))
write_ultralytics_yaml(DST / "data.yaml", DST, names)
(ROOT / "docs" / "auditoria" / "ip102_limpo_manifesto.json").write_text(
    json.dumps({"train": sorted(train), "val": sorted(val), "test": sorted(test), "removidos": removidos,
                "regra": "dHash<=3, union-find; teste oficial intacto; membros de treino/val de grupos com teste removidos; val<->train mistos vão para treino"},
               indent=1), encoding="utf-8")
print("layout:", layout)
