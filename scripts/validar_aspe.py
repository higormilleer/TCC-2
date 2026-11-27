#!/usr/bin/env python
"""Validação funcional do ASPE (Seção 6.5 do plano): YOLOv8s-ASPE e YOLO26s-ASPE.

Verifica: carga de pesos pré-treinados, camadas alteradas (entradas/saídas/canais), parâmetros e GFLOPs,
forward, backward com gradientes nos módulos ASPE, atualização dos parâmetros ASPE por um passo do
otimizador e salvar/recarregar. Escreve docs/auditoria/validacao_funcional_aspe.md.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402
from ultralytics.utils import DEFAULT_CFG  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from tcc.models.yolo_aspe import C2fASPE, convert_backbone  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
lines = ["# Validação funcional do ASPE", "", f"Ambiente: torch {torch.__version__}, dispositivo {dev}.", ""]


def check(name, yaml, weights, nc=12):
    L = [f"## {name}", ""]
    base = DetectionModel(yaml, nc=nc, verbose=False)
    p_base = sum(p.numel() for p in base.parameters())
    g_base = get_flops(base, 640)
    m = copy.deepcopy(base)
    # 1) pesos pré-treinados (antes da conversão, como no ASPETrainer)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    pre = (ck.get("ema") or ck["model"]).float().state_dict()
    own = m.state_dict()
    ok = {k: v for k, v in pre.items() if k in own and v.shape == own[k].shape}
    miss_shape = [k for k, v in pre.items() if k in own and v.shape != own[k].shape]
    m.load_state_dict(ok, strict=False)
    n_aspe_before = sum(isinstance(x, C2fASPE) for x in m.model)
    convert_backbone(m)
    aspe_params = {k for k in m.state_dict() if ".aspe." in k}
    new_keys = [k for k in m.state_dict() if k not in pre]
    L += [f"- Pesos pré-treinados: `{Path(weights).name}`; {len(ok)}/{len(own)} tensores do modelo base transferidos "
          f"({len(miss_shape)} descartados por forma diferente, ex.: camadas de classe do `Detect` com nc={nc}).",
          f"- Tensores novos do ASPE (sem correspondência, inicializados aleatoriamente): {len(aspe_params)}.",
          f"- Camadas convertidas: {sum(isinstance(x, C2fASPE) for x in m.model) - n_aspe_before} (índices 2, 4, 6, 8)."]
    p_new = sum(p.numel() for p in m.parameters())
    g_new = get_flops(m, 640)
    L += ["", "| | Parâmetros (M) | GFLOPs @640 |", "|---|---|---|",
          f"| {name.split('-')[0]} original | {p_base/1e6:.2f} | {g_base:.1f} |",
          f"| {name} | {p_new/1e6:.2f} | {g_new:.1f} |",
          f"| Diferença | +{(p_new-p_base)/1e6:.2f} (+{100*(p_new/p_base-1):.0f}%) | +{g_new-g_base:.1f} |", ""]
    L += ["| Camada | Tipo | Canais de entrada → saída | Tensores/parâmetros ASPE |", "|---|---|---|---|"]
    for i in (2, 4, 6, 8):
        b = m.model[i]
        L.append(f"| {i} | C2fASPE (base {type(base.model[i]).__name__}) | {b.cv1.conv.in_channels} → {b.cv2.conv.out_channels} | "
                 f"{sum(p.numel() for p in b.aspe.parameters())} |")
    # 2) forward/backward
    m = m.to(dev).train()
    m.args = SimpleNamespace(**{**vars(DEFAULT_CFG), "box": 7.5, "cls": 0.5, "dfl": 1.5})
    x = torch.randn(2, 3, 640, 640, device=dev)
    batch = {"img": torch.rand(2, 3, 640, 640, device=dev), "batch_idx": torch.tensor([0., 1.], device=dev),
             "cls": torch.tensor([[1.], [3.]], device=dev),
             "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.4, 0.6, 0.2, 0.4]], device=dev)}
    out = m(x)
    L += ["", f"- Forward (modo treino): saída do tipo `{type(out).__name__}`"
          + (f", chaves {list(out.keys())}" if isinstance(out, dict) else "") + "."]
    loss, items = m.loss(batch)
    loss.sum().backward()
    gr = {k: p.grad for k, p in m.named_parameters() if ".aspe." in k}
    n_ok = sum(g is not None and float(g.abs().sum()) > 0 for g in gr.values())
    L += [f"- Backward: loss = {float(loss.sum()):.3f}; **{n_ok}/{len(gr)}** tensores de parâmetros do ASPE receberam gradiente não nulo."]
    # 3) passo do otimizador atualiza o ASPE
    before = {k: p.detach().clone() for k, p in m.named_parameters() if ".aspe." in k}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    opt.step()
    n_changed = sum(not torch.equal(before[k], p.detach()) for k, p in m.named_parameters() if ".aspe." in k)
    L += [f"- Um passo de AdamW alterou **{n_changed}/{len(before)}** tensores do ASPE."]
    # 4) inferência e salvar/recarregar
    m.eval()
    with torch.no_grad():
        o = m(x)
    shp = [tuple(t.shape) for t in (o if isinstance(o, (list, tuple)) else [o]) if torch.is_tensor(t)]
    y = YOLO(yaml); y.model = m.cpu()
    tmp = ROOT / "runs" / "_tmp_validar.pt"
    y.save(tmp); r = YOLO(str(tmp))
    n_re = sum(isinstance(z, C2fASPE) for z in r.model.model)
    tmp.unlink()
    L += [f"- Inferência (modo avaliação): formas de saída {shp}.",
          f"- Salvar e recarregar checkpoint: {n_re} blocos ASPE recuperados.", ""]
    return L


lines += check("YOLOv8s-ASPE", "yolov8s.yaml", str(ROOT / "yolov8s.pt"))
lines += check("YOLO26s-ASPE", "yolo26s.yaml", str(ROOT / "yolo26s.pt"))
out = ROOT / "docs" / "auditoria" / "validacao_funcional_aspe.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
