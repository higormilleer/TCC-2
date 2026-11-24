#!/usr/bin/env python
"""Remove das imagens de validação/teste as quase duplicatas (dHash, Hamming <= 3) de outras fotos de outro split.

Regra: se um dos lados do par está no treino, sai o lado de avaliação (val/teste); se o par é val x teste, sai o da
validação (o teste permanece intacto). Os arquivos NÃO são apagados: vão para _excluidas_quase_duplicatas/.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcc.data.yolo_io import list_images  # noqa: E402

BASE = ROOT / "data" / "processed" / "agropest12_paper_grouped"
OUT = BASE / "_excluidas_quase_duplicatas"
src_key = lambda s: re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", s)
rows = [(sp, p) for sp in ("train", "val", "test") for p in list_images(BASE / "images" / sp)]


def dhash(p):
    with Image.open(p) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if v else "0" for v in (g[:, 1:] > g[:, :-1]).ravel()), 2)


H = np.array([dhash(p) for _, p in rows], dtype=np.uint64)
pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
sp = np.array([r[0] for r in rows]); key = np.array([src_key(r[1].stem) for r in rows])
pairs = []
for i in range(len(H)):
    d = pop[np.bitwise_xor(H[i + 1:], H[i]).view(np.uint8).reshape(-1, 8)].sum(axis=1)
    for j in np.nonzero(d <= 3)[0] + i + 1:
        if sp[i] != sp[j] and key[i] != key[j]:
            pairs.append((i, j))

drop = {}
for i, j in pairs:
    a, b = (rows[i][0], i), (rows[j][0], j)
    if "train" in (a[0], b[0]):
        victim = b if a[0] == "train" else a
    else:  # val x test: sai a validação
        victim = a if a[0] == "val" else b
    drop[victim[1]] = (rows[i][1].stem, rows[i][0], rows[j][1].stem, rows[j][0])

report = []
for idx, info in sorted(drop.items()):
    s, p = rows[idx]
    for sub, src in (("images", p), ("labels", BASE / "labels" / s / f"{p.stem}.txt")):
        dst = OUT / sub / s / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    report.append({"removida": p.stem, "do_split": s, "par": list(info)})
for c in (BASE / "labels").glob("*.cache"):
    c.unlink()  # caches do Ultralytics (regeneráveis)
(OUT / "relatorio.json").write_text(json.dumps({"n_pares": len(pairs), "n_imagens_removidas": len(report), "itens": report}, indent=1), encoding="utf-8")
print(f"pares encontrados: {len(pairs)} | imagens removidas: {len(report)}")
from collections import Counter
print("removidas por split:", dict(Counter(r["do_split"] for r in report)))
