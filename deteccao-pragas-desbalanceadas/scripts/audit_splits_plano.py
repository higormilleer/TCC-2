#!/usr/bin/env python
"""Etapa B do plano operacional: tabela por classe/split (imagens e instâncias) e auditoria de vazamento.

Gera docs/auditoria/split_<nome>.md e .csv para cada split do AgroPest-12 e um resumo de vazamento.
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcc.data.yolo_io import list_images, load_yaml  # noqa: E402

SPLITS = {
    "kaggle_oficial": "agropest12",
    "paper_70_15_15": "agropest12_paper",
    "paper_agrupado_70_15_15": "agropest12_paper_grouped",
    "plano_final_13143": "agropest12_plano",
}
src_key = lambda stem: re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return chr(10).join(out)


def run(tag, folder):
    base = ROOT / "data" / "processed" / folder
    names = load_yaml(base / "data.yaml")["names"]
    names = [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)
    img = {s: Counter() for s in ("train", "val", "test")}
    inst = {s: Counter() for s in img}
    n_img = Counter()
    src_split = defaultdict(set)
    for sp in img:
        for im in list_images(base / "images" / sp):
            n_img[sp] += 1
            src_split[src_key(im.stem)].add(sp)
            lab = base / "labels" / sp / f"{im.stem}.txt"
            cls = [int(l.split()[0]) for l in lab.read_text().split("\n") if l.strip()] if lab.exists() else []
            for c in set(cls):
                img[sp][c] += 1
            for c in cls:
                inst[sp][c] += 1
    rows = []
    for i, n in enumerate(names):
        rows.append({"Classe": n, "Imagens treino": img["train"][i], "Instâncias treino": inst["train"][i],
                     "Instâncias validação": inst["val"][i], "Instâncias teste": inst["test"][i],
                     "Total de instâncias": sum(inst[s][i] for s in inst)})
    df = pd.DataFrame(rows)
    tot = {c: int(df[c].sum()) for c in df.columns[1:]}
    df.loc[len(df)] = {"Classe": "Total", **tot}
    df.to_csv(ROOT / "docs" / "auditoria" / f"split_{tag}.csv", index=False, encoding="utf-8")
    leaked = sum(len(v) > 1 for v in src_split.values())
    summary = {"split": tag, "imagens treino/val/teste": f"{n_img['train']}/{n_img['val']}/{n_img['test']}",
               "total de imagens": sum(n_img.values()), "fotos originais únicas": len(src_split),
               "fotos com cópias em mais de um split": leaked}
    md = [f"# Split `{tag}`", "", *(f"- {k}: **{v}**" for k, v in summary.items()), "", md_table(df)]
    (ROOT / "docs" / "auditoria" / f"split_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    return summary


res = [run(t, f) for t, f in SPLITS.items() if (ROOT / "data" / "processed" / f).exists()]
print(pd.DataFrame(res).to_string(index=False))
