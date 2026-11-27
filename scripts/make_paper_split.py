#!/usr/bin/env python
"""Cria o split do artigo YOLO-ASPE (Sayni et al., 2025) a partir das 13.143 imagens do AgroPest-12.

O artigo diz: treino 70% (9.200), validação 15% (1.971), teste 15% (1.972), "stratified sampling".

  --mode paper          sorteio estratificado por imagem (o que o artigo descreve)
  --mode paper_grouped  idem, mas mantendo juntas as cópias aumentadas da mesma foto

ATENÇÃO (integridade do TCC): este conjunto mistura imagens que, no split do TCC, são o HOLDOUT.
Modelos treinados aqui NÃO podem ser avaliados no holdout do TCC. Servem só para a reprodução.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # noqa: E402

from tcc.data.yolo_io import dump_yaml, link_or_copy, list_images, load_yaml, read_yolo_label, write_ultralytics_yaml  # noqa: E402

SRC = ROOT / "data" / "processed" / "agropest12"


def source_key(stem: str) -> str:
    return re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem)


def split_two(Y, n_test, seed):
    sss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=n_test / len(Y), random_state=seed)
    a, b = next(sss.split(np.zeros(len(Y)), Y))
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "paper_grouped"], default="paper")
    ap.add_argument("--seed", type=int, default=42)
    ns = ap.parse_args()

    names = load_yaml(SRC / "data.yaml")["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    nc = len(names)

    items = []  # (img_path, label_path, stem)
    for sp in ("train", "val", "test"):
        for im in list_images(SRC / "images" / sp):
            items.append((im, SRC / "labels" / sp / f"{im.stem}.txt", im.stem))
    n = len(items)
    n_test, n_val = int(round(0.15 * n)), int(round(0.15 * n)) - 1  # 1972 / 1971, como no artigo
    n_train = n - n_test - n_val
    print(f"{n} imagens -> treino {n_train} / val {n_val} / teste {n_test}")

    def class_vec(lab):
        v = np.zeros(nc, dtype=int)
        for r in read_yolo_label(lab):
            v[r[0]] = 1
        return v

    if ns.mode == "paper":
        Y = np.stack([class_vec(lab) for _, lab, _ in items])
        trval, test = split_two(Y, n_test, ns.seed)
        tr_sub, va_sub = split_two(Y[trval], n_val, ns.seed + 1)
        idx = {"train": trval[tr_sub], "val": trval[va_sub], "test": test}
    else:
        groups = defaultdict(list)
        for i, (_, _, stem) in enumerate(items):
            groups[source_key(stem)].append(i)
        keys = list(groups)
        Yg = np.stack([np.maximum.reduce([class_vec(items[i][1]) for i in groups[k]]) for k in keys])
        sizes = np.array([len(groups[k]) for k in keys])
        # estratifica os GRUPOS; as proporções de imagens ficam aproximadas
        gt = int(round(0.15 * len(keys)))
        trval_g, test_g = split_two(Yg, gt, ns.seed)
        tr_g, va_g = split_two(Yg[trval_g], gt, ns.seed + 1)
        sel = lambda gi: np.array([i for g in gi for i in groups[keys[g]]])
        idx = {"train": sel(trval_g[tr_g]), "val": sel(trval_g[va_g]), "test": sel(test_g)}

    dst = ROOT / "data" / "processed" / f"agropest12_{ns.mode}"
    if dst.exists():
        raise SystemExit(f"{dst} já existe; apague a pasta antes de recriar.")
    where = {}
    for sp, ii in idx.items():
        for i in ii:
            im, lab, stem = items[i]
            link_or_copy(im, dst / "images" / sp / im.name)
            (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / sp / f"{stem}.txt").write_bytes(lab.read_bytes())
            where[stem] = sp
    write_ultralytics_yaml(dst / "data.yaml", dst, names)

    src_sets = defaultdict(set)
    for stem, sp in where.items():
        src_sets[source_key(stem)].add(sp)
    leaked = sum(len(v) > 1 for v in src_sets.values())
    info = {
        "mode": ns.mode, "seed": ns.seed,
        "n_train": len(idx["train"]), "n_val": len(idx["val"]), "n_test": len(idx["test"]),
        "unique_source_photos": len(src_sets),
        "source_photos_spanning_multiple_splits": int(leaked),
        "warning": "contém imagens do HOLDOUT do TCC; não avaliar modelos daqui no holdout do TCC",
    }
    dump_yaml(dst / "split_info.yaml", info)
    print(info)


if __name__ == "__main__":
    main()
