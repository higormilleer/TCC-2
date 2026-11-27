#!/usr/bin/env python
"""Investiga a queda da validação no fim do E05 seed 43 do IP102 (fitness 0,385 -> 0,153 nas épocas 116-126, com a loss de treino normal).

Hipótese: estouro do fp16 na validação feita DURANTE o treino (o Ultralytics valida o EMA em half). Teste: avaliar o mesmo
last_fp32.pt (EMA da época 126) na VALIDAÇÃO em fp32 e em fp16. Se fp32 ficar perto de 0,38 e fp16 perto de 0,15, é estouro
numérico (o modelo está bom); se os dois caírem, a queda é real. Também mede os maiores valores de BatchNorm (limite fp16 = 65504).

Uso:  python scripts/checar_colapso.py [run ...]        (padrão: ip102_limpo_E04_s43_ares)
      python scripts/checar_colapso.py --esperar        espera o rodar_seeds.py terminar antes (não disputa a GPU)
Grava runs/val_eval/colapso_<run>.json. O TESTE não é usado.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VAL_JSON = ROOT / "runs" / "val_eval" / "plano_val.json"
ULTIMO = "ip102_limpo_E04_s44_ares"  # último treino+avaliação do rodar_seeds.py


def esperar():
    t0 = time.time()
    while True:
        if VAL_JSON.exists() and ULTIMO in json.loads(VAL_JSON.read_text(encoding="utf-8")):
            print("rodar_seeds.py terminou; esperando 5 min para liberar a GPU", flush=True)
            time.sleep(300)
            return
        recentes = [p.stat().st_mtime for p in (ROOT / "runs" / "plano").glob("*/results.csv")]
        if recentes and time.time() - max(recentes) > 3 * 3600 and time.time() - t0 > 3 * 3600:
            sys.exit("Nenhum treino atualizou results.csv há 3 h e o último run não foi avaliado: rodar_seeds.py parou?")
        time.sleep(600)


def curva(run):
    rows = list(csv.DictReader(open(ROOT / "runs" / "plano" / run / "results.csv", encoding="utf-8")))
    k50 = next(k for k in rows[0] if "mAP50(B)" in k)
    k95 = next(k for k in rows[0] if "mAP50-95(B)" in k)
    return {int(float(r["epoch"])): 0.1 * float(r[k50]) + 0.9 * float(r[k95]) for r in rows}


def main(runs):
    import torch

    from tcc.models.yolo_aspe import load_fp32

    for run in runs:
        info = json.loads(json.dumps(__import__("yaml").safe_load(open(ROOT / "runs" / "logs" / f"plano_{run}.yaml", encoding="utf-8"))))
        w = ROOT / "runs" / "plano" / run / "weights"
        cv = curva(run)
        res = {"curva_treino_ultimas": {e: round(f, 4) for e, f in list(cv.items())[-12:]}}
        sd = torch.load(w / "last_fp32.pt", map_location="cpu", weights_only=False)
        res["epoca_last_fp32"] = sd.get("epoch")
        st = sd["state_dict"]
        rv = {k: float(v.max()) for k, v in st.items() if k.endswith("running_var")}
        top = sorted(rv.items(), key=lambda kv: -kv[1])[:5]
        res["bn_running_var_max"] = top
        res["bn_acima_limite_fp16"] = sum(v > 65504 for v in rv.values())
        res["peso_abs_max"] = max(float(v.abs().max()) for v in st.values() if v.is_floating_point())
        for nome in ("last_fp32.pt", "best_fp32.pt"):
            y, _ = load_fp32(w / nome, device="cuda")
            for half in (False, True):
                m = y.val(data=info["data"], split="val", half=half, plots=False, workers=0, verbose=False)
                f = 0.1 * float(m.box.map50) + 0.9 * float(m.box.map)
                k = f"{nome}_{'fp16' if half else 'fp32'}"
                res[k] = {"map50": float(m.box.map50), "map50_95": float(m.box.map), "fitness": f}
                print(f"{run} {k:20s} fitness {f:.4f}", flush=True)
        a, b = res["last_fp32.pt_fp32"]["fitness"], res["last_fp32.pt_fp16"]["fitness"]
        ult = list(cv.values())[-1]
        if a > 0.9 * max(cv.values()) and b < a - 0.05:
            res["conclusao"] = "estouro do fp16 na validação durante o treino; o modelo em fp32 está bom"
        elif a < 0.9 * max(cv.values()):
            res["conclusao"] = "queda real do modelo (aparece também em fp32)"
        else:
            res["conclusao"] = "inconclusivo: fp16 e fp32 parecidos e altos; a queda não se reproduz fora do treino"
        res["fitness_registrado_no_treino_ultima_epoca"] = round(ult, 4)
        print(f"{run}: {res['conclusao']} | BN acima de 65504: {res['bn_acima_limite_fp16']} | maiores: {top[:3]}", flush=True)
        (ROOT / "runs" / "val_eval" / f"colapso_{run}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--esperar" in sys.argv:
        esperar()
    main(args or ["ip102_limpo_E04_s43_ares"])
