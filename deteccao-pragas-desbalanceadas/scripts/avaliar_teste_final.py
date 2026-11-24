#!/usr/bin/env python
"""AVALIAÇÃO FINAL NO TESTE (única). Lista e regras: configs/teste_final.yaml.

Avalia o best_fp32.pt de cada run no split de TESTE em fp32, grava métricas globais, por classe e (IP102) por faixa de
frequência, e aplica o NorCal só com o γ escolhido na validação. Grava runs/val_eval/teste_final.json.
Recusa-se a rodar se o teste já foi avaliado (o arquivo existe e está completo). Se cair no meio, retoma só os que faltam.
Uso: python scripts/avaliar_teste_final.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcc.data.yolo_io import load_yaml  # noqa: E402
from tcc.models.yolo_aspe import load_fp32  # noqa: E402

# reaproveita o NorCal da validação (mesma fórmula; o patch não faz nada enquanto ESTADO["w"] é None)
_spec = importlib.util.spec_from_file_location("norcal_val", ROOT / "scripts" / "norcal_val.py")
norcal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(norcal)

CFG = load_yaml(ROOT / "configs" / "teste_final.yaml")
DEST = ROOT / "runs" / "val_eval" / "teste_final.json"
FAIXAS = load_yaml(ROOT / "configs" / "minoritarias_ip102_limpo.yaml")["faixas_de_frequencia"]


def faixas(pc, names):
    out = {}
    for f, cls in FAIXAS.items():
        v = [pc[names[c]]["ap50_95"] for c in cls if names[c] in pc]
        v50 = [pc[names[c]]["ap50"] for c in cls if names[c] in pc]
        out[f] = {"ap50_95": float(np.mean(v)) if v else None, "ap50": float(np.mean(v50)) if v50 else None, "n_classes": len(v)}
    return out


def avaliar(run, gamma=None):
    info = load_yaml(ROOT / "runs" / "logs" / f"plano_{run}.yaml")
    y, ck = load_fp32(Path(info["best"]).with_name("best_fp32.pt"), device="cuda")
    names = y.model.names
    if gamma is not None:
        n, _ = norcal.contagem_treino(info["data"], len(names))
        norcal.ESTADO["w"] = torch.tensor(n ** (-gamma), dtype=torch.float32)
    try:
        m = y.val(data=info["data"], split="test", half=False, plots=False, workers=0, verbose=False)
    finally:
        norcal.ESTADO["w"] = None
    pc = {names[int(c)]: {"ap50": float(a50), "ap50_95": float(a), "precisao": float(p), "recall": float(r)}
          for c, a50, a, p, r in zip(m.box.ap_class_index, m.box.ap50, m.box.ap, m.box.p, m.box.r)}
    r = {"split": "test", "epoca_do_melhor": ck["epoch"], "norcal_gamma": gamma,
         "map50": float(m.box.map50), "map50_95": float(m.box.map),
         "fitness": 0.1 * float(m.box.map50) + 0.9 * float(m.box.map),
         "precision": float(m.box.mp), "recall": float(m.box.mr), "por_classe": pc}
    if "ip102" in run:
        r["faixas"] = faixas(pc, names)
    return r


def main():
    tarefas = [(r, None, r) for r in CFG["agropest12"] + CFG["ip102"]]
    tarefas += [(r, float(g), f"{r}__norcal_g{g}") for r, g in CFG["norcal"].items()]
    out = json.loads(DEST.read_text(encoding="utf-8")) if DEST.exists() else {}
    if out and all(k in out for _, _, k in tarefas):
        sys.exit(f"O teste já foi avaliado ({DEST}). Ele é usado uma única vez; nada foi rodado.")
    for run, g, chave in tarefas:
        if chave in out:
            continue
        out[chave] = avaliar(run, g)
        DEST.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        x = out[chave]
        print(f"ROW {chave}: mAP50 {x['map50']:.4f} mAP50-95 {x['map50_95']:.4f} fitness {x['fitness']:.4f}", flush=True)
    print("TESTE FINAL CONCLUÍDO", flush=True)


if __name__ == "__main__":
    main()
