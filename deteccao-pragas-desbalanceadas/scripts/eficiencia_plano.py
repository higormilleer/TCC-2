#!/usr/bin/env python
"""Métricas de eficiência dos modelos do plano (Seção 6 do plano operacional): E01–E04.

Mede, para cada modelo (best_fp32.pt), com o modelo FUNDIDO (Conv+BN), como na inferência do Ultralytics:
  - parâmetros: do modelo de treino (não fundido) e do modelo fundido (inferência)
  - GFLOPs @ imgsz (fundido)
  - tamanho do arquivo best.pt (fp16, Ultralytics) e do state_dict em fp32
  - latência com batch 1: só o modelo ("modelo") e ponta a ponta com pós-processamento ("completo");
    mediana, média, p95 e FPS = 1000 / mediana(completo)
  - memória de GPU de pico na inferência (batch 1)

O YOLOv8s precisa de NMS; o YOLO26s é end-to-end (sem NMS). Por isso a latência "completo" inclui o NMS
quando a saída não já vem decodificada em (N, 300, 6).

Uso (GPU livre! a latência é invalidada se houver treino ou outro programa usando a GPU):
    python scripts/eficiencia_plano.py                         # E01–E04 na GPU
    python scripts/eficiencia_plano.py plano_E03_s42 --device cpu --iters 20   # só para testar
Grava runs/val_eval/eficiencia_plano.json e imprime a tabela em markdown.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tcc.models.yolo_aspe import load_fp32  # noqa: E402

DEFAULT = ["plano_E01_s42", "plano_E02_s42", "plano_E03_s42", "plano_E04_s42"]


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def pos_processar(out, conf=0.25, iou=0.7, max_det=300):
    """Pós-processamento: NMS se a saída for (N, 4+nc, anchors); nada se já for (N, max_det, 6)."""
    from ultralytics.utils.nms import non_max_suppression

    pred = out[0] if isinstance(out, (list, tuple)) else out
    if pred.ndim == 3 and pred.shape[-1] == 6:
        return pred, False
    return non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=max_det), True


def medir(fn, dev, warmup, iters):
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = []
    for _ in range(iters):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {
        "mediana_ms": round(statistics.median(ts), 3),
        "media_ms": round(statistics.fmean(ts), 3),
        "p95_ms": round(ts[int(0.95 * (len(ts) - 1))], 3),
        "desvio_ms": round(statistics.pstdev(ts), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=DEFAULT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--half", action="store_true", help="mede também em fp16 (GPU)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "val_eval" / "eficiencia_plano.json"))
    ns = ap.parse_args()

    from ultralytics.utils.torch_utils import get_flops

    dev = torch.device(ns.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA indisponível")
    torch.backends.cudnn.benchmark = False
    res = {"ambiente": {
        "torch": torch.__version__, "device": ns.device,
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "imgsz": ns.imgsz, "batch": 1, "warmup": ns.warmup, "iters": ns.iters,
        "nms": "conf 0.25, iou 0.7, max_det 300 (somente se a saída não for end-to-end)",
        "data": time.strftime("%Y-%m-%d %H:%M"),
    }, "modelos": {}}
    if dev.type == "cuda":
        try:
            u = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip()
            res["ambiente"]["gpu_antes_de_medir"] = u
            print("GPU antes de medir (utilização, memória):", u, "<- deve estar ~0% e sem treino rodando")
        except Exception:
            pass

    # (nome do experimento, modo): YOLOv8s só tem NMS externo; o YOLO26s é medido nos dois modos do Ultralytics
    jobs = []
    for name in ns.names:
        wdir = ROOT / "runs" / "plano" / name / "weights"
        ck0 = torch.load(str(wdir / "best_fp32.pt"), map_location="cpu", weights_only=False)
        jobs.append((name, False))
        if ck0["yaml"].get("end2end"):
            jobs.append((name, True))
        del ck0

    for name, e2e in jobs:
        wdir = ROOT / "runs" / "plano" / name / "weights"
        y, ck = load_fp32(wdir / "best_fp32.pt", "cpu")
        model = y.model.eval()
        p_treino = n_params(model)
        if hasattr(model.model[-1], "end2end"):
            model.model[-1].end2end = e2e  # True: cabeça one2one sem NMS; False: cabeça one2many + NMS externo
        model = copy.deepcopy(model).fuse(verbose=False).eval()
        p_fund = n_params(model)
        modo = "end2end (sem NMS)" if e2e else "NMS externo"
        key = f"{name} [{modo}]" if ck["yaml"].get("end2end") else name
        gflops = get_flops(model, ns.imgsz)
        pt = wdir / "best.pt"
        x = torch.randn(1, 3, ns.imgsz, ns.imgsz)

        model = model.to(dev).float()
        xd = x.to(dev)
        with torch.inference_mode():
            out = model(xd)
            pos, usou_nms = pos_processar(out)
            forma = tuple((out[0] if isinstance(out, (list, tuple)) else out).shape)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            lat_modelo = medir(lambda: model(xd), dev, ns.warmup, ns.iters)
            lat_completo = medir(lambda: pos_processar(model(xd)), dev, ns.warmup, ns.iters)
            mem = round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev.type == "cuda" else None
            r = {
                "aspe": bool(ck["aspe"]), "epoca_do_best": ck["epoch"],
                "params_treino_M": round(p_treino / 1e6, 3), "params_fundido_M": round(p_fund / 1e6, 3),
                "gflops_fundido": round(gflops, 2),
                "tamanho_best_pt_MB": round(pt.stat().st_size / 2**20, 2) if pt.exists() else None,
                "tamanho_fp32_MB": round(p_fund * 4 / 2**20, 2),
                "saida_forma": forma, "usa_nms": usou_nms,
                "latencia_modelo_fp32": lat_modelo, "latencia_completa_fp32": lat_completo,
                "fps_fp32": round(1000 / lat_completo["mediana_ms"], 1),
                "mem_pico_inferencia_MB": mem,
            }
            if ns.half and dev.type == "cuda":
                mh = copy.deepcopy(model).half()
                xh = xd.half()
                lm = medir(lambda: mh(xh), dev, ns.warmup, ns.iters)
                lc = medir(lambda: pos_processar(mh(xh)), dev, ns.warmup, ns.iters)
                r["latencia_modelo_fp16"], r["latencia_completa_fp16"] = lm, lc
                r["fps_fp16"] = round(1000 / lc["mediana_ms"], 1)
                del mh
        r["modo"] = modo
        res["modelos"][key] = r
        print(f"{key}: {r['params_treino_M']} M (treino) | {r['gflops_fundido']} GFLOPs | "
              f"modelo {lat_modelo['mediana_ms']} ms | completo {lat_completo['mediana_ms']} ms | {r['fps_fp32']} FPS | NMS={usou_nms}")
        del model, y
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n| Modelo | Params treino (M) | Params fundido (M) | GFLOPs | best.pt (MB) | Lat. modelo (ms) | Lat. completa (ms) | FPS | Mem. pico (MB) | NMS |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, r in res["modelos"].items():
        print(f"| {n} | {r['params_treino_M']} | {r['params_fundido_M']} | {r['gflops_fundido']} | {r['tamanho_best_pt_MB']} | "
              f"{r['latencia_modelo_fp32']['mediana_ms']} | {r['latencia_completa_fp32']['mediana_ms']} | {r['fps_fp32']} | "
              f"{r['mem_pico_inferencia_MB']} | {'sim' if r['usa_nms'] else 'não'} |")
    print("\nSalvo em", ns.out)


if __name__ == "__main__":
    main()
