#!/usr/bin/env python
"""Monta as divisões de treino, validação e teste usadas no TCC a partir das listas em splits/.

Uso:
    python scripts/aplicar_divisoes.py agropest12   # data/processed/agropest12 -> data/processed/agropest12_plano
    python scripts/aplicar_divisoes.py ip102        # data/processed/ip102_v2   -> data/processed/ip102_limpo
    python scripts/aplicar_divisoes.py ip102 --destino OUTRA_PASTA

Pré-requisito: python scripts/preparar_dados.py converter.
As listas reproduzem exatamente as divisões do TCC (sem vazamento entre os subconjuntos). Ao final, o script confere
o resultado com os manifestos gravados quando as divisões foram criadas (docs/auditoria/manifesto_split_plano.json
e docs/auditoria/ip102_limpo_manifesto.json). As imagens são ligadas por hard link quando possível (sem ocupar espaço
extra) e copiadas caso contrário.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.data.prepare import _materialize_yolo_layout, load_class_names  # noqa: E402
from tcc.data.yolo_io import link_or_copy, list_images, load_yaml, write_ultralytics_yaml  # noqa: E402

SPLITS = ("train", "val", "test")


def ler_lista(base: str, split: str) -> list[str]:
    linhas = (ROOT / "splits" / base / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in linhas if x.strip()]


def agropest12(destino: Path) -> None:
    origem = ROOT / "data" / "processed" / "agropest12"
    if not (origem / "data.yaml").exists():
        raise SystemExit(f"{origem} não encontrado; rode antes: python scripts/preparar_dados.py converter")
    # índice de todas as imagens convertidas (a divisão do TCC redistribui as 13.143 imagens)
    indice = {}
    for sp in SPLITS:
        for p in list_images(origem / "images" / sp):
            indice[p.name] = (p, origem / "labels" / sp / f"{p.stem}.txt")
    faltando = []
    for sp in SPLITS:
        for nome in ler_lista("agropest12", sp):
            if nome not in indice:
                faltando.append(nome)
                continue
            img, lab = indice[nome]
            link_or_copy(img, destino / "images" / sp / nome)
            (destino / "labels" / sp).mkdir(parents=True, exist_ok=True)
            (destino / "labels" / sp / f"{img.stem}.txt").write_bytes(lab.read_bytes())
    if faltando:
        raise SystemExit(f"{len(faltando)} imagens da lista não existem em {origem} (ex.: {faltando[:3]})")
    nomes = (ROOT / "splits" / "agropest12" / "classes.txt").read_text(encoding="utf-8").split("\n")
    write_ultralytics_yaml(destino / "data.yaml", destino, [n for n in nomes if n])
    # conferência com o manifesto da divisão (hash SHA-256 da lista ordenada de nomes)
    man = json.loads((ROOT / "docs" / "auditoria" / "manifesto_split_plano.json").read_text(encoding="utf-8"))
    for sp in SPLITS:
        nm = sorted(q.name for q in (destino / "images" / sp).iterdir())
        ok = len(nm) == man[sp]["n"] and hashlib.sha256("\n".join(nm).encode()).hexdigest() == man[sp]["sha256_da_lista"]
        print(f"{sp}: {len(nm)} imagens | confere com o manifesto: {'sim' if ok else 'NÃO'}")


def ip102(destino: Path) -> None:
    origem = ROOT / "data" / "processed" / "ip102_v2"
    if not (origem / "images_all").exists():
        raise SystemExit(f"{origem} não encontrado; rode antes: python scripts/preparar_dados.py converter")
    ids = {sp: ler_lista("ip102", sp) for sp in SPLITS}
    _materialize_yolo_layout(destino, origem / "images_all", origem / "labels_all", ids["train"], ids["val"], ids["test"])
    nomes = load_class_names(ROOT / "configs" / "ip102_classes.txt")
    write_ultralytics_yaml(destino / "data.yaml", destino, nomes)
    man = json.loads((ROOT / "docs" / "auditoria" / "ip102_limpo_manifesto.json").read_text(encoding="utf-8"))
    for sp in SPLITS:
        st = sorted(q.stem for q in (destino / "images" / sp).iterdir())
        print(f"{sp}: {len(st)} imagens | confere com o manifesto: {'sim' if st == sorted(man[sp]) else 'NÃO'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", choices=["agropest12", "ip102"])
    ap.add_argument("--destino", default=None, help="pasta de saída (padrão: data/processed/agropest12_plano ou ip102_limpo)")
    ns = ap.parse_args()
    padrao = {"agropest12": "agropest12_plano", "ip102": "ip102_limpo"}[ns.base]
    destino = Path(ns.destino) if ns.destino else ROOT / "data" / "processed" / padrao
    if destino.exists() and any(destino.iterdir()):
        raise SystemExit(f"{destino} já existe e não está vazio; apague antes de recriar.")
    (agropest12 if ns.base == "agropest12" else ip102)(destino)


if __name__ == "__main__":
    main()
