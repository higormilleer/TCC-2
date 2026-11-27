#!/usr/bin/env python
"""Seeds extras (43 e 44) de E03 e E05 no AgroPest-12 e no IP102, avaliação na VALIDAÇÃO e análise com 3 seeds.
Regras registradas antes dos treinos: configs/seed2.yaml.

Uso:  python scripts/rodar_seeds.py              # treina o que falta (na ordem do yaml), avalia e gera o relatório
      python scripts/rodar_seeds.py --analisar   # só refaz o relatório com o que já terminou
Pode ser interrompido e rodado de novo: retoma do last.pt e pula o que já terminou. O TESTE não é usado.
Relatório: docs/auditoria/seeds_resultados.md
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

S = load_yaml(ROOT / "configs" / "seed2.yaml")
VAL_JSON = ROOT / "runs" / "val_eval" / "plano_val.json"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
DATASETS = {"AgroPest-12": "", "IP102": "ip102_limpo"}


def nome(modelo, seed, ds):
    pre = ds if ds else "plano"
    return f"{pre}_E03_s{seed}" if modelo == "E03" else f"{pre}_E04_s{seed}_ares"


def args_treino(modelo, seed, ds):
    a = ["scripts/treinar_plano.py", "E03" if modelo == "E03" else "E04", "--seed", str(seed)]
    if modelo == "E05":
        a.append("--aspe-res")
    if ds:
        a += ["--dataset", ds]
    return a


def rodar(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=ENV)
    if r.returncode != 0:
        sys.exit(f"Falhou ({r.returncode}): {' '.join(cmd)}. Rode este script de novo para retomar.")


def concluido(run):
    return (ROOT / "runs" / "logs" / f"plano_{run}.yaml").exists()


def avaliado(run):
    return VAL_JSON.exists() and run in json.loads(VAL_JSON.read_text(encoding="utf-8"))


def treinar_tudo():
    for seed in S["seeds_novas"]:
        for ds in DATASETS.values():
            for modelo in ("E03", "E05"):
                run = nome(modelo, seed, ds)
                rodar(args_treino(modelo, seed, ds))
                if not avaliado(run):
                    rodar(["scripts/avaliar_plano.py", "val", run])


def fitness(m):
    return 0.1 * m["map50"] + 0.9 * m["map50_95"]


def veredito(deltas, limiar):
    if not deltas:
        return "sem dados"
    if len(deltas) < 2:
        return "insuficiente (só 1 seed)"
    if all(d > 0 for d in deltas) and np.mean(deltas) > limiar:
        return "E05 SUPERA E03"
    if all(d < 0 for d in deltas):
        return "E05 NÃO supera E03"
    return "não conclusivo" if any(d > 0 for d in deltas) and any(d < 0 for d in deltas) else f"positivo, mas média ≤ {limiar}"


def relatorio():
    V = json.loads(VAL_JSON.read_text(encoding="utf-8"))
    C = load_yaml(ROOT / "configs" / "minoritarias_ip102_limpo.yaml")
    names = load_yaml(ROOT / "data" / "processed" / "ip102_limpo" / "data.yaml")["names"]
    raras = [c for c in C["faixas_de_frequencia"]["raras_menos_de_20"]]
    out = ["# E03 × E05 com várias seeds — VALIDAÇÃO (fp32)\n",
           "Gerado por `scripts/rodar_seeds.py`. Regras registradas antes dos treinos em `configs/seed2.yaml`.\n"]
    for dsn, ds in DATASETS.items():
        linhas, deltas = [], []
        for seed in S["seeds_total"]:
            a, b = nome("E03", seed, ds), nome("E05", seed, ds)
            if a in V and b in V:
                fa, fb = fitness(V[a]), fitness(V[b])
                deltas.append(fb - fa)
                linhas.append(f"| {seed} | {V[a]['map50']:.4f} / {V[a]['map50_95']:.4f} / {fa:.4f} | "
                              f"{V[b]['map50']:.4f} / {V[b]['map50_95']:.4f} / {fb:.4f} | {fb - fa:+.4f} |")
        out += [f"## {dsn}\n", "| seed | E03 mAP50 / mAP50-95 / fitness | E05 mAP50 / mAP50-95 / fitness | Δ fitness |",
                "|---|---|---|---|", *linhas]
        if deltas:
            out.append(f"\nSeeds completas: {len(deltas)}. Δ médio {np.mean(deltas):+.4f} (desvio {np.std(deltas, ddof=1) if len(deltas) > 1 else float('nan'):.4f}). "
                       f"**Veredito (regra geral): {veredito(deltas, 0.005)}**\n")
        if ds == "ip102_limpo":
            dr, por_classe = [], {}
            for seed in S["seeds_total"]:
                a, b = nome("E03", seed, ds), nome("E05", seed, ds)
                if a in V and b in V:
                    pa, pb = V[a]["por_classe"], V[b]["por_classe"]
                    cl = [c for c in raras if names[c] in pa and names[c] in pb]
                    ra = float(np.mean([pa[names[c]]["ap50_95"] for c in cl]))
                    rb = float(np.mean([pb[names[c]]["ap50_95"] for c in cl]))
                    dr.append(rb - ra)
                    for c in cl:
                        por_classe.setdefault(c, []).append(pb[names[c]]["ap50_95"] - pa[names[c]]["ap50_95"])
                    out.append(f"- seed {seed}: AP50-95 raras E03 {ra:.3f} → E05 {rb:.3f} (Δ {rb - ra:+.3f})")
            if dr:
                sempre_acima = sum(1 for v in por_classe.values() if len(v) == len(dr) and all(x > 0 for x in v))
                sempre_abaixo = sum(1 for v in por_classe.values() if len(v) == len(dr) and all(x < 0 for x in v))
                out.append(f"\nClasses raras com E05 acima em TODAS as {len(dr)} seeds: {sempre_acima}/{len(por_classe)}; "
                           f"abaixo em todas: {sempre_abaixo}. Δ médio da faixa {np.mean(dr):+.3f}. "
                           f"**Veredito (classes raras): {veredito(dr, 0.02).replace('E03', 'E03 nas raras')}**")
                out.append("\nLimite: as seeds medem a variação do treino; a faixa rara tem só 32 imagens de validação. "
                           "A confirmação final é no teste (75 imagens dessas classes), uma única vez.\n")
    dest = ROOT / "docs" / "auditoria" / "seeds_resultados.md"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n" + "\n".join(out), flush=True)
    print("Gravado em", dest)


if __name__ == "__main__":
    if "--analisar" not in sys.argv:
        treinar_tudo()
    relatorio()
