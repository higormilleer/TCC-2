#!/usr/bin/env python
"""Reprodução do YOLO-ASPE (Sayni et al., 2025) no AgroPest-12, com o protocolo do artigo.

Uso:
    python scripts/repro_yolo_aspe.py baseline --seed 42   # YOLOv8s original
    python scripts/repro_yolo_aspe.py aspe --seed 42       # YOLOv8s + C2f-ASPE
    python scripts/repro_yolo_aspe.py baseline --scratch   # sem pesos pré-treinados
    python scripts/repro_yolo_aspe.py aspe --split paper   # split 70/15/15 do artigo (padrão)

Protocolo (Seção "Training Protocol" do artigo): AdamW (beta1 0.937, wd 5e-4), warmup 3 épocas
+ cosseno 0,01 -> 0,0001, batch 16, até 300 épocas com paciência 50, 640 px, mosaic até a
época 250, mixup 0,1, HSV 0,015/0,7/0,4, flips h/v, rotação +-15, escala 0,8-1,2,
perspectiva 0,0001, pesos de loss 7.5/0.5/1.5.

Diferenças inevitáveis (documentadas em docs/METHOD_ADAPTATIONS.md): split do nosso projeto
(11.502/1.095/546, em vez dos 70/15/15 do artigo), 1 GPU em vez de 2, seleção de checkpoint pela
fitness padrão do Ultralytics (0,1 mAP50 + 0,9 mAP50-95) em vez de só mAP50, e avaliação no
conjunto de VALIDAÇÃO (o holdout continua intocado).
Retoma sozinho de last.pt; pula o que já terminou.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from tcc.data.yolo_io import dump_yaml, load_yaml  # noqa: E402
from tcc.models.yolo_aspe import ASPETrainer, FP32Trainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["baseline", "aspe"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scratch", action="store_true", help="sem pesos pré-treinados (o artigo não informa)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument(
        "--split",
        choices=["ours", "paper", "paper_grouped"],
        default="paper",
        help="ours = split do TCC; paper = 70/15/15 do artigo; paper_grouped = idem sem vazar cópias da mesma foto",
    )
    ap.add_argument(
        "--exp",
        default=None,
        help="só com --split ours: treina nos dados aumentados de um experimento do TCC (ex.: E1 = Copy-Paste, o melhor cenário)",
    )
    ns = ap.parse_args()
    if ns.exp and ns.split != "ours":
        ap.error("--exp só vale com --split ours")

    tag = ("scratch" if ns.scratch else "pt") + ("" if ns.split == "ours" else f"_{ns.split}")
    name = f"yolov8s_{ns.variant}_{tag}{('_' + ns.exp) if ns.exp else ''}_s{ns.seed}"
    data_dir = "agropest12" if ns.split == "ours" else f"agropest12_{ns.split}"
    data_yaml = ROOT / "data" / "processed" / data_dir / ("experiments/" + ns.exp + "/data.yaml" if ns.exp else "data.yaml")
    if not data_yaml.exists():
        raise SystemExit(f"não achei {data_yaml}")
    run_dir = ROOT / "runs" / "repro" / name
    last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"
    done = ROOT / "runs" / "logs" / f"repro_{name}.yaml"
    if done.exists() and best.exists():
        print(f"{name} já concluído, pulando.")
        return

    kwargs = dict(
        data=str(data_yaml),
        epochs=ns.epochs, patience=50, imgsz=640, batch=16,
        optimizer="AdamW", lr0=0.01, lrf=0.01, cos_lr=True, warmup_epochs=3,
        momentum=0.937, weight_decay=0.0005,
        close_mosaic=50, mosaic=1.0, mixup=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0, scale=0.2, flipud=0.5, fliplr=0.5, perspective=0.0001,
        box=7.5, cls=0.5, dfl=1.5,
        workers=2, amp=True, seed=ns.seed, deterministic=True, plots=False,
        project=str(ROOT / "runs" / "repro"), name=name, exist_ok=False, save_period=25,
    )
    trainer = ASPETrainer if ns.variant == "aspe" else FP32Trainer

    resuming = False
    if last.exists():
        ck = torch.load(str(last), map_location="cpu", weights_only=False)
        resuming = ck.get("optimizer") is not None and ck.get("epoch", -1) >= 0
        if not resuming:
            raise RuntimeError(f"{last} é checkpoint finalizado sem log de conclusão; verifique {run_dir}.")
    t0 = time.time()
    if resuming:
        print(f"Retomando {name} de {last}")
        model = YOLO(str(last))
        results = model.train(resume=True, **({"trainer": trainer} if trainer else {}))
    else:
        model = YOLO("yolov8s.yaml" if ns.scratch else "yolov8s.pt")
        results = model.train(**kwargs, **({"trainer": trainer} if trainer else {}))
    dump_yaml(done, {
        "name": name, "variant": ns.variant, "seed": ns.seed, "split": ns.split, "data": str(data_yaml), "exp": ns.exp, "pretrained": not ns.scratch,
        "seconds": round(time.time() - t0, 1), "save_dir": str(results.save_dir),
        "best": str(Path(results.save_dir) / "weights" / "best.pt"),
        "results_csv": str(Path(results.save_dir) / "results.csv"),
    })
    print("concluído:", name)


if __name__ == "__main__":
    main()
