from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from tcc.paths import RAW, ROOT


IP102_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo"
AGROPEST_KAGGLE = "rupankarmajumdar/crop-pests-dataset"
IP102_KAGGLE_MIRROR = "rtlmhjbn/ip02-dataset"


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)


def download_agropest12() -> Path:
    dest = RAW / "agropest12"
    dest.mkdir(parents=True, exist_ok=True)
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError(
            "Kaggle CLI não encontrado. Instale com `pip install kaggle` e coloque "
            "kaggle.json em %USERPROFILE%\\.kaggle\\kaggle.json "
            f"(dataset: https://www.kaggle.com/datasets/{AGROPEST_KAGGLE})"
        )
    _run([kaggle, "datasets", "download", "-d", AGROPEST_KAGGLE, "-p", str(dest), "--unzip"])
    return dest


def download_ip102() -> Path:
    dest = RAW / "ip102"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as e:
        raise RuntimeError("pip install gdown") from e
    print("Baixando pasta oficial do Google Drive (IP102 v1.1). Use o subconjunto Detection.")
    gdown.download_folder(IP102_DRIVE_FOLDER, output=str(dest), quiet=False, remaining_ok=True)
    # descompactar tars/zips
    for archive in list(dest.rglob("*.tar")) + list(dest.rglob("*.zip")) + list(dest.rglob("*.tar.gz")):
        _extract(archive, dest)
    return dest


def _extract(archive: Path, dest: Path) -> None:
    print("extract", archive)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    shutil.unpack_archive(archive, dest)


def print_manual_instructions() -> None:
    print(
        """
Downloads oficiais
==================
AgroPest-12 (já vem com train/valid/test em YOLO):
  https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset

IP102 detecção (Wu et al., CVPR 2019) — Drive oficial:
  https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo
  Baixe JPEGImages + Annotations (+ ImageSets/Main/test.txt e train.txt se existirem).
  Espelho Kaggle (classificação, nome 'ip02' é typo): https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset

Coloque os arquivos em:
  data/raw/agropest12/
  data/raw/ip102/
""".strip()
    )
