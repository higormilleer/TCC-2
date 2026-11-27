#!/usr/bin/env python
"""IP102 (cauda longa): roda os 4 treinos em sequência e compara tudo na VALIDAÇÃO.

  1) E05 base   = YOLO26s-ASPE residual (sem augmentation extra)      -> ip102_limpo_E04_s42_ares
  2) RFS        = E05 + Repeat Factor Sampling (LVIS)                  -> ip102_limpo_E04_s42_ares_RFS
  3) CP         = E05 + copy-paste das classes raras                   -> ip102_limpo_E04_s42_ares_CP
  4) RFS+CP     = E05 + as duas                                        -> ip102_limpo_E04_s42_ares_RFS_CP
  (+ MIX = E05 + MixUp 0,3, logo após a base -> ip102_limpo_E04_s42_ares_MIX)
  No fim: NorCal (scripts/norcal_val.py, sem treino) no E03 e no melhor modelo da série.

Parâmetros e hipóteses: configs/minoritarias_ip102_limpo.yaml (registrados antes dos treinos).
Uso:  python scripts/rodar_ip102_minoritarias.py
Pode ser interrompido e rodado de novo: retoma do last.pt e pula o que já terminou. O TESTE não é usado.
Resultado: docs/auditoria/ip102_minoritarias_resultados.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcc.data.yolo_io import load_yaml  # noqa: E402

DS = "ip102_limpo"
C = load_yaml(ROOT / "configs" / f"minoritarias_{DS}.yaml")
NAMES = load_yaml(ROOT / "data" / "processed" / DS / "data.yaml")["names"]
VAL_JSON = ROOT / "runs" / "val_eval" / "plano_val.json"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
LIMIAR = 0.005
BASE = f"{DS}_E04_s42_ares"
TREINOS = [  # (rótulo, argumentos extras, nome do run)
    ("E05 base (ASPE residual)", [], BASE),
    ("E05 + MixUp 0,3", ["--mm", "MIX"], BASE + "_MIX"),
    ("E05 + RFS", ["--mm", "RFS"], BASE + "_RFS"),
    ("E05 + CP", ["--mm", "CP"], BASE + "_CP"),
    ("E05 + RFS+CP", ["--mm", "RFS+CP"], BASE + "_RFS_CP"),
]
REFS = [("E03 YOLO26s", f"{DS}_E03_s42"), ("E04 YOLO26s-ASPE (artigo)", f"{DS}_E04_s42")]


def rodar(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=ENV)
    if r.returncode != 0:
        sys.exit(f"Falhou ({r.returncode}): {' '.join(cmd)}. Rode este script de novo para retomar.")


def metricas(run):
    val = json.loads(VAL_JSON.read_text(encoding="utf-8")) if VAL_JSON.exists() else {}
    if run not in val:
        rodar(["scripts/avaliar_plano.py", "val", run])
        val = json.loads(VAL_JSON.read_text(encoding="utf-8"))
    m = val[run]
    m["fitness"] = 0.1 * m["map50"] + 0.9 * m["map50_95"]
    pc = m["por_classe"]
    for faixa, classes in C["faixas_de_frequencia"].items():  # média do AP das classes da faixa presentes na val
        a = [pc[NAMES[c]]["ap50_95"] for c in classes if NAMES[c] in pc]
        b = [pc[NAMES[c]]["ap50"] for c in classes if NAMES[c] in pc]
        m[faixa] = (float(np.mean(a)) if a else float("nan"), float(np.mean(b)) if b else float("nan"), len(a))
    return m


def main():
    for rot, extra, run in TREINOS:
        print(f"\n===== {rot}: {run} =====", flush=True)
        rodar(["scripts/treinar_plano.py", "E04", "--aspe-res", "--dataset", DS, *extra])
        m = metricas(run)
        print(f"== {rot}: mAP50 {m['map50']:.4f} mAP50-95 {m['map50_95']:.4f} fitness {m['fitness']:.4f}", flush=True)

    linhas = [(r, n, metricas(n)) for r, n in REFS] + [(r, n, metricas(n)) for r, _, n in TREINOS]
    base = metricas(BASE)
    fx = list(C["faixas_de_frequencia"])
    cab = ("| Modelo | mAP50 | mAP50-95 | fitness | Δ vs E05 | " + " | ".join(f"AP50-95 {f.split('_')[0]} (AP50)" for f in fx)
           + " | época do melhor |")
    out = ["# IP102 (cauda longa): ASPE residual + augmentation das classes raras — VALIDAÇÃO (fp32)\n",
           "Gerado por `scripts/rodar_ip102_minoritarias.py`. Parâmetros/hipóteses: `configs/minoritarias_ip102_limpo.yaml`.",
           "Faixas pelo nº de imagens de TREINO por classe (raras < 20, comuns 20–99, frequentes ≥ 100); entre parênteses, nº de "
           "classes da faixa presentes na val. Seed única; limiar de ruído 0,005 de fitness.\n",
           cab, "|" + "---|" * (cab.count("|") - 1)]
    for rot, run, m in linhas:
        out.append(f"| {rot} | {m['map50']:.4f} | {m['map50_95']:.4f} | {m['fitness']:.4f} | {m['fitness'] - base['fitness']:+.4f} | "
                   + " | ".join(f"{m[f][0]:.3f} ({m[f][1]:.3f}; n={m[f][2]})" for f in fx) + f" | {m['epoca_do_melhor']} |")
    melhor = max(linhas, key=lambda t: t[2]["fitness"])
    out.append(f"\nMaior fitness na validação (sem pós-processamento): **{melhor[0]}** ({melhor[1]}).")

    # NorCal (sem treino) no E03 e no melhor modelo treinado nesta série
    melhor_treinado = max([t for t in linhas if t[1] not in dict(REFS).values()], key=lambda t: t[2]["fitness"])
    alvos_nc = [REFS[0][1], melhor_treinado[1]]
    faltam = [r for r in alvos_nc if not (ROOT / "runs" / "val_eval" / f"norcal_{r}.json").exists()]
    if faltam:
        rodar(["scripts/norcal_val.py", *faltam])
    out += ["\n## NorCal (recalibração pós-treino pela frequência no treino; γ escolhido NA VALIDAÇÃO, otimista)\n",
            "| Modelo | γ | mAP50 | mAP50-95 | fitness | Δ vs sem NorCal | AP50-95 raras / comuns / frequentes |", "|---|---|---|---|---|---|---|"]
    for r in alvos_nc:
        R = json.loads((ROOT / "runs" / "val_eval" / f"norcal_{r}.json").read_text(encoding="utf-8"))
        for k in ("sem_norcal", R["melhor_gamma_na_val"]):
            x = R[k]
            out.append(f"| {r} | {k} | {x['map50']:.4f} | {x['map50_95']:.4f} | {x['fitness']:.4f} | "
                       f"{x['fitness'] - R['sem_norcal']['fitness']:+.4f} | " + " / ".join(f"{v:.3f}" for v in x["faixas_ap50_95"].values()) + " |")
    dest = ROOT / "docs" / "auditoria" / "ip102_minoritarias_resultados.md"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n" + "\n".join(out), flush=True)
    print("\nGravado em", dest)


if __name__ == "__main__":
    main()
