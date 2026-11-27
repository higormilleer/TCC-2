from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

from tcc.data.yolo_io import dump_yaml, link_or_copy, list_images, write_ultralytics_yaml, write_yolo_label, xyxy_to_xywhn
from tcc.paths import PROCESSED, ROOT, SPLITS, dataset_processed, dataset_raw


def load_class_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # "1 rice leaf roller" or just "Ants"
        parts = line.split(maxsplit=1)
        if parts[0].isdigit() and len(parts) == 2:
            names.append(parts[1].strip())
        else:
            names.append(line)
    return names


def voc_xml_to_yolo(xml_path: Path, names: list[str], name_to_id: dict[str, int]) -> tuple[str, int, int, list[tuple[int, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = (root.findtext("filename") or xml_path.stem).strip()
    size = root.find("size")
    width = int(float(size.findtext("width") or 0))
    height = int(float(size.findtext("height") or 0))
    rows = []
    for obj in root.findall("object"):
        raw = (obj.findtext("name") or "").strip()
        cls_id = None
        if raw.isdigit():
            idx = int(raw)
            # IP102: o <name> do XML é índice 0-BASED (0..101; existem caixas com 0 e nenhuma com 102).
            # A versão anterior tratava 1..102 como 1-based e subtraía 1, deslocando todas as classes
            # (XML 0 e 1 viravam o mesmo id e o id 101 ficava vazio). Corrigido em 04/11/2026; ver
            # docs/auditoria/ip102_mapeamento_de_classes.md. Os dados já gerados NÃO foram regerados.
            if 0 <= idx < len(names):
                cls_id = idx
        else:
            key = raw.lower()
            cls_id = name_to_id.get(key)
            if cls_id is None:
                # fuzzy: strip extra spaces
                cls_id = name_to_id.get(" ".join(key.split()))
        if cls_id is None:
            continue
        b = obj.find("bndbox")
        xmin = float(b.findtext("xmin"))
        ymin = float(b.findtext("ymin"))
        xmax = float(b.findtext("xmax"))
        ymax = float(b.findtext("ymax"))
        if xmax <= xmin or ymax <= ymin:
            continue
        if width <= 0 or height <= 0:
            continue
        xc, yc, w, h = xyxy_to_xywhn(xmin, ymin, xmax, ymax, width, height)
        if w <= 0 or h <= 0:
            continue
        rows.append((cls_id, xc, yc, w, h))
    return filename, width, height, rows


def _find_split_lists(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in ("train.txt", "trainval.txt", "val.txt", "valid.txt", "test.txt"):
        hits = list(root.rglob(name))
        if hits:
            found[name] = hits[0]
    return found


def _read_id_list(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\\", "/")
        if not line:
            continue
        ids.append(Path(line).stem)
    return ids


def prepare_ip102(names: list[str] | None = None, out_name: str = "ip102") -> Path:
    """Converte VOC → YOLO e monta holdout oficial (test) + val estratificada do train.

    out_name: nome da pasta de saída em data/processed e data/splits (use 'ip102_v2' para os dados com o mapeamento corrigido,
    sem sobrescrever a versão antiga 'ip102', que tem o bug de classes).
    """
    raw = dataset_raw("ip102")
    out = dataset_processed(out_name)
    names_path = ROOT / "configs" / "ip102_classes.txt"
    names = names or load_class_names(names_path)
    name_to_id = {n.lower(): i for i, n in enumerate(names)}

    jpeg = next((p for p in raw.rglob("JPEGImages") if p.is_dir()), None)
    ann = next((p for p in raw.rglob("Annotations") if p.is_dir()), None)
    if jpeg is None or ann is None:
        raise FileNotFoundError(
            "IP102: esperado JPEGImages/ e Annotations/ em data/raw/ip102. "
            "Baixe o Detection do Drive oficial (xpwu95/IP102)."
        )

    img_index = {p.stem: p for p in list_images(jpeg)}
    xmls = sorted(ann.glob("*.xml"))
    if not xmls:
        xmls = sorted(ann.rglob("*.xml"))

    labels_all = out / "labels_all"
    images_all = out / "images_all"
    labels_all.mkdir(parents=True, exist_ok=True)
    images_all.mkdir(parents=True, exist_ok=True)

    converted = []
    skipped_xml: list[str] = []
    for xml_path in tqdm(xmls, desc="VOC→YOLO IP102"):
        try:
            filename, width, height, rows = voc_xml_to_yolo(xml_path, names, name_to_id)
        except ET.ParseError as e:
            skipped_xml.append(f"{xml_path.name}: {e}")
            continue
        stem = Path(filename).stem
        src = img_index.get(stem) or img_index.get(xml_path.stem)
        if src is None or not rows:
            continue
        if width <= 0 or height <= 0:
            im = cv2.imread(str(src))
            if im is None:
                continue
            height, width = im.shape[:2]
            rows = []
            # re-parse with real size
            filename, _, _, rows = voc_xml_to_yolo(xml_path, names, name_to_id)
            # boxes were normalized with xml size 0 — redo using actual pixels from xml
            tree = ET.parse(xml_path)
            rows = []
            for obj in tree.getroot().findall("object"):
                raw_name = (obj.findtext("name") or "").strip()
                # índice 0-BASED também aqui (mesma correção de voc_xml_to_yolo; o '- 1' antigo repetia o bug)
                cls_id = (int(raw_name) if 0 <= int(raw_name) < len(names) else None) if raw_name.isdigit() else name_to_id.get(raw_name.lower())
                if cls_id is None:
                    continue
                b = obj.find("bndbox")
                xc, yc, w, h = xyxy_to_xywhn(
                    float(b.findtext("xmin")),
                    float(b.findtext("ymin")),
                    float(b.findtext("xmax")),
                    float(b.findtext("ymax")),
                    width,
                    height,
                )
                rows.append((cls_id, xc, yc, w, h))
        dst_img = images_all / src.name
        if not dst_img.exists():
            link_or_copy(src, dst_img)
        write_yolo_label(labels_all / f"{src.stem}.txt", rows)
        converted.append(src.stem)

    lists = _find_split_lists(raw)
    official_test = set(_read_id_list(lists["test.txt"])) if "test.txt" in lists else set()
    if "train.txt" in lists:
        official_train = set(_read_id_list(lists["train.txt"]))
    elif "trainval.txt" in lists:
        official_train = set(_read_id_list(lists["trainval.txt"])) - official_test
    else:
        official_train = set(converted) - official_test

    converted_set = set(converted)
    train_ids = sorted(official_train & converted_set)
    holdout_ids = sorted(official_test & converted_set)
    leftover = sorted(converted_set - set(train_ids) - set(holdout_ids))
    train_ids.extend(leftover)

    from tcc.data.splits import stratified_train_val

    train_final, val_ids = stratified_train_val(
        images_all, labels_all, train_ids, val_fraction=0.15 / 0.85, seed=42, n_classes=len(names)
    )

    layout = _materialize_yolo_layout(out, images_all, labels_all, train_final, val_ids, holdout_ids)
    write_ultralytics_yaml(out / "data.yaml", out, names)
    dump_yaml(
        SPLITS / out_name / "split_info.yaml",
        {
            "dataset": out_name,
            "source": "official detection test as holdout; val stratified from official train",
            "paper_split": "Wu et al. CVPR 2019: 15178 train / 3798 test (sem val oficial de detecção)",
            "n_converted": len(converted),
            "n_train": len(train_final),
            "n_val": len(val_ids),
            "n_holdout": len(holdout_ids),
            "seed": 42,
            "official_test_file": str(lists.get("test.txt", "")),
            "n_xml_total": len(xmls),
            "n_xml_skipped_malformed": len(skipped_xml),
            "xml_skipped_examples": skipped_xml[:20],
        },
    )
    _write_manifest(out_name, train_final, val_ids, holdout_ids, names)
    if skipped_xml:
        (SPLITS / out_name / "skipped_xml.txt").write_text("\n".join(skipped_xml) + "\n", encoding="utf-8")
    return layout


def prepare_agropest12() -> Path:
    raw = dataset_raw("agropest12")
    out = dataset_processed("agropest12")
    # Roboflow-style: train/valid/test with images+labels, or a data.yaml
    yaml_hits = list(raw.rglob("data.yaml"))
    train_dir = next((p for p in raw.rglob("train") if p.is_dir() and (p / "images").exists() or list((p).glob("*"))), None)

    # Prefer explicit folders
    def find_split(names: tuple[str, ...]) -> Path | None:
        for n in names:
            for p in raw.rglob(n):
                if p.is_dir():
                    imgs = p / "images" if (p / "images").exists() else p
                    if list_images(imgs):
                        return p
        return None

    train_p = find_split(("train",))
    val_p = find_split(("valid", "val", "validation"))
    test_p = find_split(("test", "holdout"))

    names = [
        "Ants",
        "Bees",
        "Beetles",
        "Caterpillars",
        "Earthworms",
        "Earwigs",
        "Grasshoppers",
        "Moths",
        "Slugs",
        "Snails",
        "Wasps",
        "Weevils",
    ]
    if yaml_hits:
        import yaml

        y = yaml.safe_load(yaml_hits[0].read_text(encoding="utf-8"))
        if isinstance(y.get("names"), dict):
            names = [y["names"][k] for k in sorted(y["names"], key=lambda x: int(x))]
        elif isinstance(y.get("names"), list):
            names = list(y["names"])

    if train_p and val_p and test_p:
        import shutil

        out.mkdir(parents=True, exist_ok=True)
        mapping = {"train": train_p, "val": val_p, "test": test_p}
        ids = {}
        for split, src in mapping.items():
            img_src = src / "images" if (src / "images").exists() else src
            lab_src = src / "labels" if (src / "labels").exists() else src
            img_dst = out / "images" / split
            lab_dst = out / "labels" / split
            img_dst.mkdir(parents=True, exist_ok=True)
            lab_dst.mkdir(parents=True, exist_ok=True)
            split_ids = []
            for im in list_images(img_src):
                dst = img_dst / im.name
                if not dst.exists():
                    link_or_copy(im, dst)
                lab = lab_src / f"{im.stem}.txt"
                if lab.exists():
                    (lab_dst / lab.name).write_bytes(lab.read_bytes())
                split_ids.append(im.stem)
            ids[split] = split_ids
        write_ultralytics_yaml(out / "data.yaml", out, names)
        dump_yaml(
            SPLITS / "agropest12" / "split_info.yaml",
            {
                "dataset": "agropest12",
                "source": "pastas train/valid/test do Kaggle (holdout reaproveitado)",
                "n_train": len(ids["train"]),
                "n_val": len(ids["val"]),
                "n_holdout": len(ids["test"]),
                "seed": None,
                "method": "vendor_split",
            },
        )
        _write_manifest("agropest12", ids["train"], ids["val"], ids["test"], names)
        return out

    # Fallback: pasta única → divisão estratificada 70/15/15
    images = list_images(raw)
    if not images:
        raise FileNotFoundError("AgroPest-12 não encontrado em data/raw/agropest12")
    labels_guess = raw / "labels"
    from tcc.data.splits import make_holdout_split

    return make_holdout_split("agropest12", images, labels_guess if labels_guess.exists() else raw, names, seed=42)


def _materialize_yolo_layout(out: Path, images_all: Path, labels_all: Path, train_ids, val_ids, test_ids) -> Path:
    import shutil

    id_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    img_index = {p.stem: p for p in list_images(images_all)}
    for split, ids in id_map.items():
        img_dst = out / "images" / split
        lab_dst = out / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lab_dst.mkdir(parents=True, exist_ok=True)
        for stem in ids:
            src = img_index.get(stem)
            if src is None:
                continue
            dst = img_dst / src.name
            if not dst.exists():
                link_or_copy(src, dst)
            lab = labels_all / f"{stem}.txt"
            if lab.exists():
                (lab_dst / lab.name).write_bytes(lab.read_bytes())
    return out


def _write_manifest(dataset: str, train_ids, val_ids, test_ids, names: list[str]) -> None:
    dest = SPLITS / dataset
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "names": names,
        "train": list(train_ids),
        "val": list(val_ids),
        "holdout": list(test_ids),
        "overlap_train_val": sorted(set(train_ids) & set(val_ids)),
        "overlap_train_holdout": sorted(set(train_ids) & set(test_ids)),
        "overlap_val_holdout": sorted(set(val_ids) & set(test_ids)),
    }
    (dest / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for split, ids in ("train", train_ids), ("val", val_ids), ("holdout", test_ids):
        (dest / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
