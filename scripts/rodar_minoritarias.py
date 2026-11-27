#!/usr/bin/env python
"""Roda TODA a série de augmentation das classes minoritárias sobre o E05 (YOLO26s-ASPE residual), em sequência:

  1) treina as 3 variantes isoladas: RFS, CP, CPC (configs/minoritarias.yaml);
  2) avalia cada uma na VALIDAÇÃO (fp32) e compara com o E05;
  3) aplica a regra de combinação registrada no yaml (2 melhores; as 3 se todas passarem do limiar) e treina as combinações;
  4) grava a tabela final em docs/auditoria/minoritarias_resultados.md.

Uso:  python scripts/rodar_minoritarias.py
Pode ser interrompido (Ctrl+C) e rodado de novo: retoma o treino do last.pt e pula o que já terminou.
O TESTE não é usado em nenhum momento.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcc.data.yolo_io import load_yaml  # noqa: E402

C = load_yaml(ROOT / "configs" / "minoritarias.yaml")
REF = C["referencia"]
VARS = ["RFS", "CP", "CPC"]
LIMIAR = 0.005
VAL_JSON = ROOT / "runs" / "val_eval" / "plano_val.json"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
DS = ROOT / "data" / "processed" / "agropest12_plano"


def nome(vs):
    return REF + "_" + "_".join(v for v in VARS if v in vs)


def rodar(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=ENV)
    if r.returncode != 0:
        sys.exit(f"Falhou ({r.returncode}): {' '.join(cmd)}. Rode este script de novo para retomar.")


def treinar(vs):
    rodar(["scripts/treinar_plano.py", "E04", "--aspe-res", "--mm", "+".join(vs)])
    return nome(vs)


def metricas(run):
    val = json.loads(VAL_JSON.read_text(encoding="utf-8")) if VAL_JSON.exists() else {}
    if run not in val:
        rodar(["scripts/avaliar_plano.py", "val", run])
        val = json.loads(VAL_JSON.read_text(encoding="utf-8"))
    m = val[run]
    m["fitness"] = 0.1 * m["map50"] + 0.9 * m["map50_95"]
    return m


def confusoes_grupos(run):
    """Objetos com caixa certa (IoU>=0,5) e classe errada (conf>=0,25), dentro dos grupos de parceiros do CPC (VALIDAÇÃO)."""
    pj = ROOT / "runs" / "val_eval" / "pred" / run / "predictions.json"
    if not pj.exists():
        rodar(["scripts/dump_preds_val.py", run])
    from PIL import Image
    J = json.loads(pj.read_text())
    off = 0 if min(j["category_id"] for j in J) == 0 else 1
    P = {}
    for j in J:
        if j["score"] >= 0.25:
            x, y, w, h = j["bbox"]
            P.setdefault(str(j["image_id"]), []).append((j["category_id"] - off, x, y, x + w, y + h, j["score"]))
    par = {int(k): set(v) for k, v in C["parceiros"].items()}
    total = grupo = 0
    for lf in sorted((DS / "labels" / "val").glob("*.txt")):
        im = next(iter((DS / "images" / "val").glob(lf.stem + ".*")))
        W, H = Image.open(im).size
        g = []
        for l in lf.read_text().splitlines():
            p = l.split()
            if len(p) >= 5:
                c, x, y, w, h = int(float(p[0])), *map(float, p[1:5])
                g.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
        p = np.array(P.get(lf.stem, [])).reshape(-1, 6)
        if not g or not len(p):
            continue
        for c, *b in g:
            x1 = np.maximum(b[0], p[:, 1]); y1 = np.maximum(b[1], p[:, 2])
            x2 = np.minimum(b[2], p[:, 3]); y2 = np.minimum(b[3], p[:, 4])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            iou = inter / ((b[2] - b[0]) * (b[3] - b[1]) + (p[:, 3] - p[:, 1]) * (p[:, 4] - p[:, 2]) - inter + 1e-9)
            ok = iou >= 0.5
            if not ok.any() or (p[ok, 0] == c).any():
                continue
            pc = int(p[np.argmax(np.where(ok, p[:, 5], -1)), 0])
            total += 1
            grupo += int(pc in par.get(c, ()))
    return total, grupo


def main():
    ref = metricas(REF)
    print(f"\nReferência {REF}: mAP50 {ref['map50']:.4f} mAP50-95 {ref['map50_95']:.4f} fitness {ref['fitness']:.4f}", flush=True)

    # 1) isoladas
    res = {}
    for v in VARS:
        run = treinar([v])
        res[(v,)] = (run, metricas(run))
        print(f"== {v}: fitness {res[(v,)][1]['fitness']:.4f} (Δ {res[(v,)][1]['fitness'] - ref['fitness']:+.4f})", flush=True)

    # 2) regra de combinação (configs/minoritarias.yaml)
    delta = {v: res[(v,)][1]["fitness"] - ref["fitness"] for v in VARS}
    cand = sorted([v for v in VARS if delta[v] > 0], key=lambda v: -delta[v])
    combos = []
    if len(cand) >= 2:
        combos.append(tuple(v for v in VARS if v in cand[:2]))
    if all(delta[v] > LIMIAR for v in VARS):
        combos.append(tuple(VARS))
    print("\nCandidatas à combinação:", cand or "nenhuma", "| combinações a treinar:", ["+".join(c) for c in combos] or "nenhuma", flush=True)
    for cb in combos:
        run = treinar(list(cb))
        res[cb] = (run, metricas(run))

    # 3) tabela final
    alvo = ["Beetles", "Caterpillars", "Earthworms", "Slugs"]
    linhas = [("E05 (referência)", REF, ref)] + [("+".join(k), r, m) for k, (r, m) in res.items()]
    cab = "| Variante | mAP50 | mAP50-95 | fitness | Δ fitness | " + " | ".join(f"AP50 {a}" for a in alvo) + \
          " | confusões (total / nos grupos) | época do melhor |"
    out = [f"# Augmentation das classes minoritárias sobre o E05 — VALIDAÇÃO (fp32)\n",
           "Gerado por `scripts/rodar_minoritarias.py`. Regra e hipóteses: `configs/minoritarias.yaml` (registradas antes dos treinos).",
           f"Limiar de ruído: {LIMIAR} de fitness (seed única).\n", cab, "|" + "---|" * (cab.count("|") - 1)]
    for rot, run, m in linhas:
        tot, grp = confusoes_grupos(run)
        pc = m["por_classe"]
        out.append(f"| {rot} | {m['map50']:.4f} | {m['map50_95']:.4f} | {m['fitness']:.4f} | {m['fitness'] - ref['fitness']:+.4f} | "
                   + " | ".join(f"{pc[a]['ap50']:.3f}" for a in alvo) + f" | {tot} / {grp} | {m['epoca_do_melhor']} |")
    melhor = max(linhas, key=lambda t: t[2]["fitness"])
    out.append(f"\nMaior fitness na validação: **{melhor[0]}** ({melhor[1]}), Δ {melhor[2]['fitness'] - ref['fitness']:+.4f} sobre o E05."
               + ("" if melhor[2]["fitness"] - ref["fitness"] > LIMIAR else " Diferença dentro do ruído: não é conclusiva."))
    dest = ROOT / "docs" / "auditoria" / "minoritarias_resultados.md"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n" + "\n".join(out), flush=True)
    print("\nGravado em", dest)


if __name__ == "__main__":
    main()
