#!/usr/bin/env python
"""Auditoria de integridade e vazamento do split agrupado (Seção 4 do plano; itens c4 e c5)."""
from __future__ import annotations

import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcc.data.yolo_io import list_images  # noqa: E402

BASE = ROOT / "data" / "processed" / (sys.argv[1] if len(sys.argv) > 1 else "agropest12_paper_grouped")
TAG = sys.argv[2] if len(sys.argv) > 2 else "agrupado"
src_key = lambda s: re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", s)
rows = []  # (split, path, stem)
for sp in ("train", "val", "test"):
    for p in list_images(BASE / "images" / sp):
        rows.append((sp, p, p.stem))

bad_img, empty_lab, bad_lines, bad_cls, bad_box = [], [], 0, 0, 0
md5 = defaultdict(list)
dh = []
for sp, p, stem in rows:
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        dh.append(int("".join("1" if v else "0" for v in (g[:, 1:] > g[:, :-1]).ravel()), 2))
    except Exception as e:  # noqa: BLE001
        bad_img.append((str(p.name), str(e)[:40])); dh.append(0)
    md5[hashlib.md5(p.read_bytes()).hexdigest()].append((sp, stem))
    lab = BASE / "labels" / sp / f"{stem}.txt"
    lines = [l for l in lab.read_text().split("\n") if l.strip()] if lab.exists() else []
    if not lines:
        empty_lab.append(stem)
    for l in lines:
        t = l.split()
        if len(t) != 5:
            bad_lines += 1; continue
        c, x, y, w, h = int(t[0]), *map(float, t[1:])
        if not 0 <= c < 12: bad_cls += 1
        if w <= 0 or h <= 0 or x - w / 2 < -1e-3 or y - h / 2 < -1e-3 or x + w / 2 > 1 + 1e-3 or y + h / 2 > 1 + 1e-3:
            bad_box += 1

exact_cross = [v for v in md5.values() if len(v) > 1 and len({s for s, _ in v}) > 1]
exact_any = [v for v in md5.values() if len(v) > 1]
# quase duplicatas: dHash de 64 bits, Hamming <= 3, em splits diferentes e de fotos-fonte diferentes
H = np.array(dh, dtype=np.uint64)
sp_arr = np.array([r[0] for r in rows]); key_arr = np.array([src_key(r[2]) for r in rows])
pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
near = 0; examples = []
for i in range(len(H)):
    x = np.bitwise_xor(H[i + 1:], H[i]).view(np.uint8).reshape(-1, 8)
    d = pop[x].sum(axis=1)
    for j in np.nonzero(d <= 3)[0] + i + 1:
        if sp_arr[i] != sp_arr[j] and key_arr[i] != key_arr[j]:
            near += 1
            if len(examples) < 5: examples.append((rows[i][2][:35], sp_arr[i], rows[j][2][:35], sp_arr[j]))

cls_inst = {sp: Counter() for sp in ("train", "val", "test")}
for sp, p, stem in rows:
    for l in (BASE / "labels" / sp / f"{stem}.txt").read_text().split("\n"):
        if l.strip(): cls_inst[sp][int(l.split()[0])] += 1
names = ["Ants","Bees","Beetles","Caterpillars","Earthworms","Earwigs","Grasshoppers","Moths","Slugs","Snails","Wasps","Weevils"]
tot = {sp: sum(c.values()) for sp, c in cls_inst.items()}
L = [f"# Integridade e vazamento — split {TAG}", "",
     f"- Imagens: {len(rows)} | ilegíveis: **{len(bad_img)}** | labels vazios (imagem sem objeto): **{len(empty_lab)}**",
     f"- Linhas de label malformadas: **{bad_lines}** | classe fora de 0–11: **{bad_cls}** | caixas inválidas/fora da imagem: **{bad_box}**",
     f"- Duplicatas exatas (MD5): em qualquer split **{len(exact_any)}** grupos; **entre splits diferentes: {len(exact_cross)}**",
     f"- Quase duplicatas (dHash, Hamming ≤ 3) entre fotos-fonte diferentes e splits diferentes: **{near}**",
     f"- Fotos originais (grupos de cópias) em mais de um split: **0** (garantido pela construção; verificado em `audit_splits_plano.py`)", "",
     "| Classe | % das instâncias no treino | % na validação | % no teste |", "|---|---|---|---|"]
for i, n in enumerate(names):
    L.append(f"| {n} | {100*cls_inst['train'][i]/tot['train']:.1f} | {100*cls_inst['val'][i]/tot['val']:.1f} | {100*cls_inst['test'][i]/tot['test']:.1f} |")
(ROOT / "docs" / "auditoria" / f"integridade_{TAG}.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L)); print("exemplos quase-duplicatas:", examples)
