#!/usr/bin/env python
"""NorCal (Pan et al., NeurIPS 2021) adaptado ao YOLO (sigmoid), SEM retreinar: recalibra a confiança de cada classe pela sua
frequência no TREINO antes do NMS e mede na VALIDAÇÃO para vários γ. Parâmetros: configs/minoritarias_ip102_limpo.yaml.

    s'_c = max_k(s_k) · (s_c · n_c^-γ) / Σ_k (s_k · n_k^-γ)

Uso:  python scripts/norcal_val.py ip102_limpo_E03_s42 ip102_limpo_E04_s42_ares ...
Grava runs/val_eval/norcal_<run>.json e imprime a tabela. O TESTE não é usado.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

from tcc.data.yolo_io import load_yaml  # noqa: E402
from tcc.models.yolo_aspe import load_fp32  # noqa: E402

ESTADO = {"w": None}
_orig = DetectionValidator.postprocess


def _post(self, preds):
    w = ESTADO["w"]
    if w is not None:
        p = preds[0] if isinstance(preds, (list, tuple)) else preds
        nc = w.numel()
        assert p.dim() == 3 and p.shape[1] == 4 + nc, f"formato inesperado {tuple(p.shape)} (esperado (B, 4+{nc}, N))"
        s = p[:, 4:, :].float()
        ws = s * w.to(s.device)[None, :, None]
        s2 = s.max(1, keepdim=True).values * ws / ws.sum(1, keepdim=True).clamp_min(1e-12)
        p = torch.cat([p[:, :4], s2.to(p.dtype)], 1)
        preds = (p, *preds[1:]) if isinstance(preds, (list, tuple)) else p
    return _orig(self, preds)


DetectionValidator.postprocess = _post


def contagem_treino(data_yaml, nc):
    d = load_yaml(Path(data_yaml))
    lab = Path(data_yaml).parent / "labels" / "train"
    n = Counter()
    for f in lab.glob("*.txt"):
        n.update(int(float(l.split()[0])) for l in f.read_text().splitlines() if l.strip())
    return np.array([max(1, n[c]) for c in range(nc)], dtype=np.float64), d


def main():
    runs = sys.argv[1:]
    for run in runs:
        info = load_yaml(ROOT / "runs" / "logs" / f"plano_{run}.yaml")
        ds = Path(info["data"]).parent.name
        C = load_yaml(ROOT / "configs" / f"minoritarias_{ds}.yaml")
        gammas = C["pos_processamento_norcal"]["gammas"]
        y, _ = load_fp32(Path(info["best"]).with_name("best_fp32.pt"), device="cuda")
        names = y.model.names
        nc = len(names)
        n, _ = contagem_treino(info["data"], nc)
        res = {}
        for g in [None] + list(gammas):  # None = sem NorCal (referência)
            ESTADO["w"] = None if g is None else torch.tensor(n ** (-g), dtype=torch.float32)
            m = y.val(data=info["data"], split="val", half=False, plots=False, workers=0, verbose=False)
            pc = {names[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap)}
            faixas = {}
            for f, cls in C["faixas_de_frequencia"].items():
                v = [pc[names[c]] for c in cls if names[c] in pc]
                faixas[f] = float(np.mean(v)) if v else float("nan")
            k = "sem_norcal" if g is None else f"gamma_{g}"
            res[k] = {"map50": float(m.box.map50), "map50_95": float(m.box.map),
                      "fitness": 0.1 * float(m.box.map50) + 0.9 * float(m.box.map), "faixas_ap50_95": faixas}
            print(f"{run} {k:12s} mAP50 {res[k]['map50']:.4f} mAP50-95 {res[k]['map50_95']:.4f} fitness {res[k]['fitness']:.4f} | "
                  + " ".join(f"{f.split('_')[0]} {v:.3f}" for f, v in faixas.items()), flush=True)
        ESTADO["w"] = None
        melhor = max((k for k in res if k != "sem_norcal"), key=lambda k: res[k]["fitness"])
        res["melhor_gamma_na_val"] = melhor
        print(f"{run}: melhor {melhor}, Δ fitness {res[melhor]['fitness'] - res['sem_norcal']['fitness']:+.4f} (escolhido NA VAL)", flush=True)
        (ROOT / "runs" / "val_eval" / f"norcal_{run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
