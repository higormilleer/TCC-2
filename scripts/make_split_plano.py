#!/usr/bin/env python
"""Split final do plano: usa TODAS as 13.143 imagens do AgroPest-12, 70/15/15, sem vazamento.

Em vez de descartar quase-duplicatas, elas são FUNDIDAS ao mesmo grupo (union-find) antes da divisão:
  - cópias aumentadas da mesma foto (mesmo nome-base do Roboflow) -> mesmo grupo;
  - imagens quase idênticas (dHash de 64 bits, Hamming <= 3; inclui duplicatas exatas) -> mesmo grupo.
Os grupos são estratificados (iterative stratification, Sechidis et al. 2011) por classe E por faixas de
número de instâncias (>=1, >=2, >=4, >=8 por classe), para equilibrar também instâncias, não só imagens.
A seed é escolhida entre 42..51 usando apenas estatísticas dos DADOS (nunca resultados de modelo).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # noqa: E402

from tcc.data.yolo_io import dump_yaml, link_or_copy, list_images, load_yaml, write_ultralytics_yaml  # noqa: E402

SRC = ROOT / "data" / "processed" / "agropest12"
DST = ROOT / "data" / "processed" / "agropest12_plano"
THR = (1, 2, 4, 8)
src_key = lambda s: re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", s)

names = load_yaml(SRC / "data.yaml")["names"]
names = [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)
nc = len(names)
items = [(p, SRC / "labels" / sp / f"{p.stem}.txt") for sp in ("train", "val", "test") for p in list_images(SRC / "images" / sp)]
N = len(items)


def dhash(p):
    with Image.open(p) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if v else "0" for v in (g[:, 1:] > g[:, :-1]).ravel()), 2)


H = np.array([dhash(p) for p, _ in items], dtype=np.uint64)
parent = list(range(N))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra

first = {}
for i, (p, _) in enumerate(items):
    k = src_key(p.stem)
    if k in first: union(first[k], i)
    else: first[k] = i
pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
n_near = 0
for i in range(N):
    d = pop[np.bitwise_xor(H[i + 1:], H[i]).view(np.uint8).reshape(-1, 8)].sum(axis=1)
    for j in np.nonzero(d <= 3)[0] + i + 1:
        if find(i) != find(int(j)): n_near += 1
        union(i, int(j))
groups = defaultdict(list)
for i in range(N): groups[find(i)].append(i)
keys = list(groups); G = len(keys)
print(f"{N} imagens -> {G} grupos (arestas de quase-duplicata que fundiram grupos: {n_near})")

cnt = np.zeros((N, nc), dtype=int)
for i, (_, lab) in enumerate(items):
    for l in lab.read_text().split("\n"):
        if l.strip(): cnt[i, int(l.split()[0])] += 1
Gc = np.stack([cnt[groups[k]].sum(axis=0) for k in keys])           # instâncias por classe em cada grupo
Y = np.concatenate([(Gc >= t).astype(int) for t in THR], axis=1)     # presença e faixas de contagem
size = np.array([len(groups[k]) for k in keys])


def try_seed(seed):
    n_test = int(round(0.15 * G))
    a, te = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed).split(np.zeros(G), Y))
    tr, va = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed + 1).split(np.zeros(len(a)), Y[a]))
    S = {"train": a[tr], "val": a[va], "test": te}
    img = {s: int(size[ix].sum()) for s, ix in S.items()}
    inst = {s: Gc[ix].sum(axis=0) for s, ix in S.items()}
    share = {s: inst[s] / inst[s].sum() for s in S}
    dev = max(float(np.abs(share[s] - share["train"]).max()) for s in ("val", "test"))
    shares_img = {s: img[s] / N for s in img}
    ok = all(abs(shares_img[s] - t) <= 0.01 for s, t in (("train", .70), ("val", .15), ("test", .15)))
    return S, img, dev, ok


best = None
for seed in range(42, 52):
    S, img, dev, ok = try_seed(seed)
    print(f"seed {seed}: imagens {img} | maior desvio de instâncias por classe (val/teste vs treino) = {100*dev:.2f} pp | proporções OK: {ok}")
    if ok and (best is None or dev < best[3]): best = (seed, S, img, dev)
assert best, "nenhuma seed atendeu às proporções 70/15/15 ±1 pp"
seed, S, img, dev = best
print(f"seed escolhida: {seed} (desvio {100*dev:.2f} pp)")

if DST.exists(): raise SystemExit(f"{DST} já existe; apague antes de recriar.")
for sp, gi in S.items():
    for g in gi:
        for i in groups[keys[g]]:
            p, lab = items[i]
            link_or_copy(p, DST / "images" / sp / p.name)
            (DST / "labels" / sp).mkdir(parents=True, exist_ok=True)
            (DST / "labels" / sp / f"{p.stem}.txt").write_bytes(lab.read_bytes())
write_ultralytics_yaml(DST / "data.yaml", DST, names)
man = {}
for sp in ("train", "val", "test"):
    nm = sorted(q.name for q in (DST / "images" / sp).iterdir())
    man[sp] = {"n": len(nm), "sha256_da_lista": hashlib.sha256("\n".join(nm).encode()).hexdigest()}
man.update({"seed": seed, "total": N, "grupos": G, "metodo": "union-find (cópias Roboflow + dHash<=3) + iterative stratification por classe e faixas de instâncias"})
(ROOT / "docs" / "auditoria" / "manifesto_split_plano.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
dump_yaml(DST / "split_info.yaml", man)
print(json.dumps(man, indent=1, ensure_ascii=False))
