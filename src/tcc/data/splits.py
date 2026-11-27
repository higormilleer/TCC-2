from __future__ import annotations

from pathlib import Path

import numpy as np

from tcc.data.yolo_io import link_or_copy, read_yolo_label
from tcc.data.prepare import _materialize_yolo_layout, _write_manifest
from tcc.data.yolo_io import write_ultralytics_yaml
from tcc.paths import dataset_processed


def image_class_sets(images_dir: Path, labels_dir: Path, ids: list[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for stem in ids:
        rows = read_yolo_label(labels_dir / f"{stem}.txt")
        out[stem] = {r[0] for r in rows}
    return out


def stratified_train_val(
    images_dir: Path,
    labels_dir: Path,
    ids: list[str],
    val_fraction: float,
    seed: int,
    n_classes: int,
) -> tuple[list[str], list[str]]:
    """Estratificação multilabel (Sechidis, Tsoumakas & Vlahavas, 2011 — ECML PKDD),
    via `iterstrat.MultilabelStratifiedShuffleSplit`.

    A versão anterior (heurística gulosa caseira) tinha um bug real: a pontuação
    comparava a fração de uma classe *dentro* de um split com o tamanho-alvo
    *do split inteiro* — grandezas diferentes — o que sistematicamente jogava
    quase todas as instâncias de dezenas de classes inteiras para o val,
    deixando `train=0` para elas. Isso foi detectado na auditoria (54/102
    classes do IP102 com zero instâncias de treino) e corrigido usando a
    implementação de referência do algoritmo, já listada em requirements.txt.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    ids = list(ids)
    labels = image_class_sets(images_dir, labels_dir, ids)

    Y = np.zeros((len(ids), n_classes), dtype=np.int64)
    for i, stem in enumerate(ids):
        for c in labels.get(stem, ()):
            if 0 <= c < n_classes:
                Y[i, c] = 1
    X = np.zeros((len(ids), 1))

    n_val = max(1, int(round(len(ids) * val_fraction)))
    if n_val >= len(ids):
        n_val = len(ids) - 1
    test_size = n_val / len(ids)

    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, Y))

    train_ids = sorted(ids[i] for i in train_idx)
    val_ids = sorted(ids[i] for i in val_idx)
    return train_ids, val_ids


def make_holdout_split(
    dataset: str,
    images: list[Path],
    labels_dir: Path,
    names: list[str],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Path:
    from tcc.paths import PROCESSED
    import shutil

    out = dataset_processed(dataset)
    images_all = out / "images_all"
    labels_all = out / "labels_all"
    images_all.mkdir(parents=True, exist_ok=True)
    labels_all.mkdir(parents=True, exist_ok=True)
    ids = []
    for im in images:
        dst = images_all / im.name
        if not dst.exists():
            from tcc.data.yolo_io import link_or_copy

            link_or_copy(im, dst)
        lab = labels_dir / f"{im.stem}.txt"
        if lab.exists():
            (labels_all / lab.name).write_bytes(lab.read_bytes())
        ids.append(im.stem)

    holdout_frac = ratios[2]
    val_from_rest = ratios[1] / (ratios[0] + ratios[1])
    rest, holdout = stratified_train_val(images_all, labels_all, ids, holdout_frac, seed, len(names))
    train, val = stratified_train_val(images_all, labels_all, rest, val_from_rest, seed + 1, len(names))
    _materialize_yolo_layout(out, images_all, labels_all, train, val, holdout)
    write_ultralytics_yaml(out / "data.yaml", out, names)
    from tcc.data.yolo_io import dump_yaml
    from tcc.paths import SPLITS

    dump_yaml(
        SPLITS / dataset / "split_info.yaml",
        {
            "dataset": dataset,
            "source": "stratified multilabel 70/15/15",
            "n_train": len(train),
            "n_val": len(val),
            "n_holdout": len(holdout),
            "seed": seed,
            "method": "iterative_stratification_sechidis2011",
        },
    )
    _write_manifest(dataset, train, val, holdout, names)
    return out
