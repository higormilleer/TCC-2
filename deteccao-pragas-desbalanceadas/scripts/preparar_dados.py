#!/usr/bin/env python
"""Baixa e converte as duas bases para o formato YOLO (etapa anterior às divisões do TCC).

Uso:
    python scripts/preparar_dados.py baixar      # AgroPest-12 (Kaggle CLI) e IP102 (Google Drive oficial)
    python scripts/preparar_dados.py converter   # gera data/processed/agropest12 e data/processed/ip102_v2

Se o download automático falhar, o script mostra onde baixar cada base e em que pasta colocá-la
(data/raw/agropest12 e data/raw/ip102). Depois da conversão, use scripts/aplicar_divisoes.py para montar
exatamente as divisões usadas no TCC, ou scripts/make_split_plano.py e scripts/limpar_vazamento_ip102.py para
regenerá-las do zero.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def baixar() -> None:
    from tcc.data.download import download_agropest12, download_ip102, print_manual_instructions

    for nome, funcao in (("AgroPest-12", download_agropest12), ("IP102", download_ip102)):
        try:
            funcao()
        except Exception as erro:  # o download depende de credenciais e de serviços externos
            print(f"{nome}: {erro}")
            print_manual_instructions()


def converter() -> None:
    from tcc.data.prepare import prepare_agropest12, prepare_ip102

    print("AgroPest-12 ->", prepare_agropest12())
    # 'ip102_v2' é a conversão com o mapeamento de classes corrigido (docs/auditoria/ip102_mapeamento_de_classes.md)
    print("IP102 ->", prepare_ip102(out_name="ip102_v2"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["baixar", "converter"])
    ns = ap.parse_args()
    baixar() if ns.etapa == "baixar" else converter()


if __name__ == "__main__":
    main()
