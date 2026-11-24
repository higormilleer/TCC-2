#!/usr/bin/env python
"""Treina um experimento do plano operacional (E01–E04) segundo configs/protocolo_plano.yaml.

Uso:
    python scripts/treinar_plano.py E01            # YOLOv8s
    python scripts/treinar_plano.py E02            # YOLOv8s-ASPE
    python scripts/treinar_plano.py E03            # YOLO26s
    python scripts/treinar_plano.py E04            # YOLO26s-ASPE
    python scripts/treinar_plano.py E03 --epochs 5 --tag sanidade   # teste curto (não vale como resultado)
    python scripts/treinar_plano.py E03 --aug A02  # ablação de augmentation (A02 sem Mosaic/MixUp; A03 conservador; A05 sem nenhum)

Retoma sozinho de last.pt e pula o que já terminou. Grava best_fp32.pt/last_fp32.pt (usados nas avaliações).
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
from tcc.data.minoritarias import AUG_ULTRALYTICS, VARIANTES, config_combinada, trainer_minoritarias  # noqa: E402
from tcc.models.yolo_aspe import ASPEFig2Trainer, ASPEResNZTrainer, ASPEResTrainer, ASPETrainer, FP32Trainer  # noqa: E402
from tcc.models.perda import trainer_focal  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["E01", "E02", "E03", "E04"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="só para testes curtos; o protocolo é 300")
    ap.add_argument("--tag", default="", help="sufixo do nome (ex.: sanidade)")
    ap.add_argument("--lr0", type=float, default=None, help="só para o estudo de lr (configs/estudo_lr.yaml)")
    ap.add_argument("--aug", choices=["A02", "A03", "A04", "A05"], default=None,
                    help="ablação de augmentation (configs/ablacao_plano.yaml); A01 é o próprio treino de referência, não precisa rodar")
    ap.add_argument("--dataset", default=None,
                    help="outro dataset em data/processed (ex.: ip102_v2). Mesmo protocolo; nome do run começa com o dataset")
    ap.add_argument("--aspe-res", action="store_true",
                    help="variante E05 (fora do artigo): ASPE com soma residual e inicialização em zero; só com E02/E04")
    ap.add_argument("--mm", default=None,
                    help="augmentation das classes minoritárias (configs/minoritarias[_<dataset>].yaml): RFS, CP, CPC, MIX ou "
                         "combinação com '+', ex.: RFS+CP. Só com --aspe-res")
    ap.add_argument("--aspe-fig2", action="store_true",
                    help="ASPE lido pela Figura 2 do artigo (CA + CIB dentro dos bottlenecks); só com E02 (configs/aspe_fig2.yaml)")
    ap.add_argument("--perda", choices=["W", "FL"], default=None,
                    help="modificação da perda de classificação (configs/perda_classes.yaml): W = weighted loss (cls_pw), "
                         "FL = Focal Loss; só com --aspe-res e sem --mm")
    ap.add_argument("--sem-zero", action="store_true",
                    help="ablação do E05: soma residual SEM inicialização em zero (configs/aspe_res_sem_zero.yaml); só com --aspe-res")
    ns = ap.parse_args()
    if ns.sem_zero and (not ns.aspe_res or ns.mm or ns.perda or ns.aspe_fig2):
        ap.error("--sem-zero só com --aspe-res, sem --mm/--perda/--aspe-fig2")
    if ns.perda and (not ns.aspe_res or ns.mm or ns.aspe_fig2):
        ap.error("--perda só com --aspe-res, sem --mm/--aspe-fig2")
    if ns.aspe_res and ns.exp not in ("E02", "E04"):
        ap.error("--aspe-res só faz sentido com E02 ou E04")
    if ns.aspe_fig2 and (ns.exp != "E02" or ns.aspe_res or ns.mm):
        ap.error("--aspe-fig2 só com E02, sem --aspe-res/--mm")
    mm_vars = []
    if ns.mm:
        if not ns.aspe_res:
            ap.error("--mm só com --aspe-res (as variantes de minoritárias são sobre o E05)")
        pedidas = {v.strip().upper() for v in ns.mm.split("+") if v.strip()}
        if not pedidas or pedidas - set(VARIANTES):
            ap.error(f"--mm aceita {VARIANTES} separados por '+'")
        mm_vars = [v for v in VARIANTES if v in pedidas]

    P = load_yaml(ROOT / "configs" / "protocolo_plano.yaml")
    spec, tr, aug = P["experimentos"][ns.exp], P["treino"], dict(P["augmentation"])
    if ns.aug:  # só as operações listadas em 'mudancas_em_relacao_a_A01' mudam; o resto fica igual ao protocolo congelado
        A = load_yaml(ROOT / "configs" / "ablacao_plano.yaml")
        aug.update(A[ns.aug]["mudancas_em_relacao_a_A01"])
    seed = ns.seed if ns.seed is not None else tr["seed"]
    prefixo = ns.dataset if ns.dataset else "plano"
    name = f"{prefixo}_{ns.exp}_s{seed}" + (f"_{ns.aug}" if ns.aug else "") + ("_ares" if ns.aspe_res else "") + ("_nz" if ns.sem_zero else "") + ("_afig2" if ns.aspe_fig2 else "") + ("_" + "_".join(mm_vars) if mm_vars else "") + (f"_{ns.perda}" if ns.perda else "") + (f"_{ns.tag}" if ns.tag else "")
    data = (ROOT / "data" / "processed" / ns.dataset / "data.yaml") if ns.dataset else (ROOT / P["dataset"]["split"] / "data.yaml")
    run_dir = ROOT / "runs" / "plano" / name
    # exist_ok=False cria name-2, name-3... quando a pasta já existe; para retomar, usa a mais recente que tem last.pt
    cand = [run_dir] + [ROOT / "runs" / "plano" / f"{name}-{i}" for i in range(2, 30)]
    com_last = [d for d in cand if (d / "weights" / "last.pt").exists()]
    if com_last:
        run_dir = max(com_last, key=lambda d: (d / "weights" / "last.pt").stat().st_mtime)
    last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"
    done = ROOT / "runs" / "logs" / f"plano_{name}.yaml"
    if done.exists() and best.exists():
        print(f"{name} já concluído, pulando.")
        return

    epochs = ns.epochs or tr["epochs"]
    # em treinos curtos mantém-se a mesma proporção do protocolo: mosaic desligado nos últimos 50/300 das épocas
    close_mosaic = aug["close_mosaic"] if epochs == tr["epochs"] else max(1, round(epochs * aug["close_mosaic"] / tr["epochs"]))
    kwargs = dict(
        data=str(data), epochs=epochs, patience=tr["patience"] if epochs == tr["epochs"] else epochs, imgsz=tr["imgsz"], batch=tr["batch"],
        optimizer=tr["optimizer"], lr0=ns.lr0 if ns.lr0 is not None else tr["lr0"], lrf=tr["lrf"], cos_lr=tr["cos_lr"], warmup_epochs=tr["warmup_epochs"],
        momentum=0.937, weight_decay=tr["weight_decay"], amp=tr["amp"], seed=seed, deterministic=tr["deterministic"],
        workers=tr["workers"],
        close_mosaic=close_mosaic, mosaic=aug["mosaic"], mixup=aug["mixup"],
        hsv_h=aug["hsv_h"], hsv_s=aug["hsv_s"], hsv_v=aug["hsv_v"], degrees=aug["degrees"], translate=aug["translate"],
        scale=aug["scale"], perspective=aug["perspective"], fliplr=aug["fliplr"], flipud=aug["flipud"],
        box=P["loss"]["box"], cls=P["loss"]["cls"], dfl=P["loss"]["dfl"],
        erasing=0.0, auto_augment=None,  # sem efeito em detecção; explícito para constar no args.yaml
        plots=False, project=str(ROOT / "runs" / "plano"), name=name, exist_ok=False, save_period=25,
    )
    trainer = (ASPEResTrainer if ns.aspe_res else ASPETrainer) if spec["aspe"] else FP32Trainer
    if ns.sem_zero:
        trainer = ASPEResNZTrainer
        print("ablação: ASPE residual SEM inicialização em zero")
    if ns.aspe_fig2:
        trainer = ASPEFig2Trainer
    perda = None
    if ns.perda:
        PC = load_yaml(ROOT / "configs" / "perda_classes.yaml")["variantes"][ns.perda]
        if ns.perda == "W":
            kwargs["cls_pw"] = PC["cls_pw"]  # weighted loss nativa do Ultralytics 8.4
            perda = {"tipo": "W", "cls_pw": PC["cls_pw"]}
        else:
            trainer = trainer_focal(trainer, PC["gamma"], PC["alpha"])
            perda = {"tipo": "FL", "gamma": PC["gamma"], "alpha": PC["alpha"]}
        print("modificação da perda:", perda)
    mm = None
    if mm_vars:
        cfg_mm = ROOT / "configs" / (f"minoritarias_{ns.dataset}.yaml" if ns.dataset else "minoritarias.yaml")
        mm = config_combinada(load_yaml(cfg_mm), mm_vars)
        for k in AUG_ULTRALYTICS:  # ex.: MIX muda o mixup do próprio Ultralytics, não o trainer
            if k in mm:
                kwargs[k] = mm.pop(k)
        trainer = trainer_minoritarias(trainer, mm)
        print("augmentation de minoritárias:", "+".join(mm_vars), mm, "| mixup =", kwargs["mixup"])

    resuming = False
    if last.exists():
        ck = torch.load(str(last), map_location="cpu", weights_only=False)
        resuming = ck.get("optimizer") is not None and ck.get("epoch", -1) >= 0
        if not resuming:
            raise RuntimeError(f"{last} é checkpoint finalizado sem log de conclusão; verifique {run_dir}.")
    t0 = time.time()
    if resuming:
        print(f"Retomando {name} de {last}")
        results = YOLO(str(last)).train(resume=True, trainer=trainer)
    else:
        results = YOLO(str(ROOT / spec["pesos"])).train(trainer=trainer, **kwargs)
    sd = Path(results.save_dir)
    dump_yaml(done, {
        "name": name, "exp": ns.exp, "aug": ns.aug or "A01", "aspe_res": bool(ns.aspe_res), "aspe_fig2": bool(ns.aspe_fig2), "sem_zero": bool(ns.sem_zero), "minoritarias": "+".join(mm_vars) or None, "mm": mm, "perda": perda, "modelo": spec["modelo"], "aspe": spec["aspe"], "seed": seed,
        "epochs_configuradas": kwargs["epochs"], "lr0": kwargs["lr0"], "close_mosaic": kwargs["close_mosaic"], "seconds": round(time.time() - t0, 1), "data": str(data),
        "save_dir": str(sd), "best": str(sd / "weights" / "best.pt"), "results_csv": str(sd / "results.csv"),
    })
    print("concluído:", name)


if __name__ == "__main__":
    main()
