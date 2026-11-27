#!/usr/bin/env python
"""Mostra no terminal os resultados do plano (E01-E04): validação (fp32) e eficiência.

Lê runs/val_eval/plano_val.json e runs/val_eval/eficiencia_plano.json (gerados por
avaliar_plano.py e eficiencia_plano.py). Não roda nenhum modelo e não usa o teste.

Uso:  python scripts/mostrar_resultados.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOMES = {
    "plano_E01_s42": "E01 YOLOv8s",
    "plano_E02_s42": "E02 YOLOv8s-ASPE",
    "plano_E03_s42": "E03 YOLO26s",
    "plano_E04_s42": "E04 YOLO26s-ASPE",
}


def tabela(cab, linhas):
    larg = [max(len(str(x)) for x in col) for col in zip(cab, *linhas)]
    f = lambda r: "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(r, larg)))
    print(f(cab))
    print("  ".join("-" * w for w in larg))
    for r in linhas:
        print(f(r))


def main():
    pv = ROOT / "runs" / "val_eval" / "plano_val.json"
    if pv.exists():
        d = json.loads(pv.read_text(encoding="utf-8"))
        linhas, fit = [], {}
        for k, nome in NOMES.items():
            v = d.get(k)
            if not v:
                continue
            fit[k] = 0.1 * v["map50"] + 0.9 * v["map50_95"]
            linhas.append([nome, v["epoca_do_melhor"], f"{v['map50']:.4f}", f"{v['map50_95']:.4f}",
                           f"{fit[k]:.4f}", f"{v['precision']:.3f}", f"{v['recall']:.3f}"])
        print("\n=== VALIDACAO (fp32, melhor checkpoint; o teste NAO foi usado) ===")
        tabela(["Modelo", "Epoca", "mAP50", "mAP50-95", "Fitness", "P", "R"], linhas)
        print("\nFitness = 0,1*mAP50 + 0,9*mAP50-95. Diferencas < 0,005 estao dentro do ruido (1 seed).")
        pares = [("plano_E02_s42", "plano_E01_s42", "ASPE no YOLOv8s (E02 - E01)"),
                 ("plano_E04_s42", "plano_E03_s42", "ASPE no YOLO26s (E04 - E03)"),
                 ("plano_E03_s42", "plano_E01_s42", "YOLO26s vs YOLOv8s (E03 - E01)")]
        print("\nDiferencas de fitness:")
        for a, b, t in pares:
            if a in fit and b in fit:
                dif = fit[a] - fit[b]
                print(f"  {t}: {dif:+.4f}  -> {'dentro do ruido' if abs(dif) < 0.005 else 'acima do ruido'}")
    else:
        print("sem", pv)

    pe = ROOT / "runs" / "val_eval" / "eficiencia_plano.json"
    if pe.exists():
        e = json.loads(pe.read_text(encoding="utf-8"))
        linhas = []
        for k, r in e["modelos"].items():
            base = k.split(" [")[0]
            nome = NOMES.get(base, base) + (" [" + k.split(" [")[1] if " [" in k else "")
            linhas.append([nome, r["params_treino_M"], r["gflops_fundido"], r["latencia_completa_fp32"]["mediana_ms"],
                           r["fps_fp32"], r["mem_pico_inferencia_MB"], r["tamanho_best_pt_MB"]])
        print("\n=== EFICIENCIA (batch 1, 640, fp32, RTX 4060 Ti) ===")
        tabela(["Modelo", "Params(M)", "GFLOPs", "Lat.completa(ms)", "FPS", "Mem.pico(MB)", "best.pt(MB)"], linhas)
        print("\nMedido em", e["ambiente"].get("data"), "| GPU antes de medir:", e["ambiente"].get("gpu_antes_de_medir"))
    else:
        print("sem", pe)


if __name__ == "__main__":
    main()
