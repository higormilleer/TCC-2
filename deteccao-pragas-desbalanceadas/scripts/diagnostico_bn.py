#!/usr/bin/env python
"""Diagnóstico da queda da validação no fim do E05 (ASPE residual) no IP102: loss de treino normal, mas na validação a
loss de classificação explode e a de caixa não muda. Hipótese: as estatísticas de BatchNorm (running_mean/var, que o EMA
também média) se descolaram do que a rede realmente produz; em treino o BN usa as estatísticas do lote e não sente isso.

Testes, com o last_fp32.pt (época da queda) e o best_fp32.pt (referência):
 1. por camada BN: compara running_mean/var com a média/variância reais da entrada em imagens de TREINO (modo eval);
    lista as camadas mais descoladas;
 2. recalibração: zera as estatísticas de BN e as recalcula com ~800 imagens de TREINO (sem gradiente, pesos intocados)
    e avalia de novo na VALIDAÇÃO em fp32. Se o last recuperar, a queda é das estatísticas de BN, não dos pesos.
Uso: python scripts/diagnostico_bn.py [run]   (padrão: ip102_limpo_E04_s43_ares_repro). O TESTE não é usado.
Grava runs/val_eval/diagnostico_bn_<run>.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ultralytics.data.augment import LetterBox  # noqa: E402

from tcc.data.yolo_io import load_yaml  # noqa: E402
from tcc.models.yolo_aspe import load_fp32  # noqa: E402

N_IMGS, BATCH = 800, 16


def lotes(data_yaml, n=N_IMGS, seed=0):
    imgs = sorted((Path(data_yaml).parent / "images" / "train").glob("*"))
    random.Random(seed).shuffle(imgs)
    lb = LetterBox((640, 640), auto=False)
    buf = []
    for p in imgs[:n]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = lb(image=im)[:, :, ::-1].transpose(2, 0, 1)
        buf.append(torch.from_numpy(np.ascontiguousarray(im)).float() / 255)
        if len(buf) == BATCH:
            yield torch.stack(buf).cuda()
            buf = []
    if buf:
        yield torch.stack(buf).cuda()


def descolamento(model, data_yaml):
    """Em modo eval, mede média/variância reais da entrada de cada BN e compara com as running."""
    bns = {n: m for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)}
    acc = {n: [0.0, 0.0, 0] for n in bns}
    hooks = []
    for n, m in bns.items():
        def h(mod, inp, out, n=n):
            x = inp[0].detach().float()
            acc[n][0] += x.mean((0, 2, 3)) * x.shape[0]
            acc[n][1] += x.var((0, 2, 3), unbiased=False) * x.shape[0]
            acc[n][2] += x.shape[0]
        hooks.append(m.register_forward_hook(h))
    model.eval()
    with torch.no_grad():
        for x in lotes(data_yaml, n=320):
            model(x)
    for h in hooks:
        h.remove()
    res = {}
    for n, m in bns.items():
        mu, var, k = acc[n]
        if k == 0:
            continue
        mu, var = mu / k, var / k
        rv, rm = m.running_var.float(), m.running_mean.float()
        # deslocamento da média em desvios-padrão e razão de variâncias (1 = ok)
        z = ((mu - rm).abs() / (rv + m.eps).sqrt()).max().item()
        r = (var / (rv + m.eps)).max().item()
        res[n] = {"desvio_media_max_em_sd": round(z, 2), "razao_var_max": round(r, 2)}
    return dict(sorted(res.items(), key=lambda kv: -max(kv[1]["desvio_media_max_em_sd"], kv[1]["razao_var_max"]))[:10])


def recalibrar(model, data_yaml):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # média acumulada exata
    model.train()
    with torch.no_grad():
        for x in lotes(data_yaml):
            model(x)
    model.eval()


def avaliar(y, data):
    m = y.val(data=data, split="val", half=False, plots=False, workers=0, verbose=False)
    return {"map50": float(m.box.map50), "map50_95": float(m.box.map), "fitness": 0.1 * float(m.box.map50) + 0.9 * float(m.box.map)}


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "ip102_limpo_E04_s43_ares_repro"
    info = load_yaml(ROOT / "runs" / "logs" / f"plano_{run}.yaml")
    w = ROOT / "runs" / "plano" / run / "weights"
    out = {}
    for nome in ("last_fp32.pt", "best_fp32.pt"):
        r = {}
        y, _ = load_fp32(w / nome, device="cuda")
        r["epoca"] = torch.load(w / nome, map_location="cpu", weights_only=False).get("epoch")
        r["camadas_bn_mais_descoladas"] = descolamento(y.model, info["data"])
        r["val_fp32_original"] = avaliar(y, info["data"])
        y, _ = load_fp32(w / nome, device="cuda")  # recarrega (val() funde conv+BN)
        recalibrar(y.model, info["data"])
        r["val_fp32_bn_recalibrado"] = avaliar(y, info["data"])
        out[nome] = r
        print(f"\n{run} {nome} (época {r['epoca']}): fitness {r['val_fp32_original']['fitness']:.4f} -> "
              f"BN recalibrado {r['val_fp32_bn_recalibrado']['fitness']:.4f}", flush=True)
        for n, v in list(r["camadas_bn_mais_descoladas"].items())[:5]:
            print(f"   {n:45s} desvio da média {v['desvio_media_max_em_sd']:>8} sd | razão de variância {v['razao_var_max']}", flush=True)
    (ROOT / "runs" / "val_eval" / f"diagnostico_bn_{run}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
