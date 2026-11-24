#!/usr/bin/env python
"""Teste (C2): remove só a INFORMAÇÃO (posição / seletividade de canal) dos portões, mantendo a escala média (inferência, sem retreinar, VALIDAÇÃO).
 - CA sem posição: o mapa de atenção A(h,w) vira sua média espacial (por imagem e canal)
 - CA só eixo H / só eixo W: um dos dois fatores vira constante (média)
 - SE sem seletividade: o vetor s por canal vira sua média sobre canais (por imagem)
Uso: python scripts/teste_aspe_c2.py plano_E02_s42 plano_E04_s42 ..."""
import sys, types
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from tcc.data.yolo_io import load_yaml
from tcc.models.yolo_aspe import load_fp32, ASPEUnit

def ca_fwd(mode):
    def f(self, x):
        _, _, h, w = x.shape
        f_h = x.mean(dim=3, keepdim=True); f_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        g = self.act(self.bn1(self.conv1(torch.cat([f_h, f_w], dim=2))))
        g_h, g_w = torch.split(g, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(g_h)); a_w = torch.sigmoid(self.conv_w(g_w.permute(0, 1, 3, 2)))
        if mode == "sem_posicao":
            A = (a_h * a_w).mean(dim=(2, 3), keepdim=True); return x * A
        if mode == "so_H": return x * a_h * a_w.mean(dim=3, keepdim=True)
        if mode == "so_W": return x * a_h.mean(dim=2, keepdim=True) * a_w
        return x * a_h * a_w
    return f
def se_fwd(mode):
    def f(self, x):
        s = x.mean(dim=(2, 3), keepdim=True); s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        if mode == "sem_seletividade": s = s.mean(dim=1, keepdim=True)
        return x * s
    return f
VARS = [("completo", None, None), ("CA sem posição", "sem_posicao", None), ("CA só eixo H", "so_H", None), ("CA só eixo W", "so_W", None), ("SE sem seletividade", None, "sem_seletividade")]
for name in sys.argv[1:]:
    info = load_yaml(ROOT / "runs" / "logs" / f"plano_{name}.yaml"); print(f"\n== {name}")
    for lab, cm, sm in VARS:
        y, _ = load_fp32(Path(info["best"]).with_name("best_fp32.pt"), device="cuda")
        for m in y.model.modules():
            if isinstance(m, ASPEUnit):
                if cm: m.ca.forward = types.MethodType(ca_fwd(cm), m.ca)
                if sm: m.se.forward = types.MethodType(se_fwd(sm), m.se)
        r = y.val(data=info["data"], split="val", half=False, plots=False, workers=0, verbose=False)
        ap = np.asarray(r.box.all_ap).mean(0)
        print(f"  {lab:<20} mAP50 {r.box.map50:.4f} | mAP50-95 {r.box.map:.4f} | AP50 {ap[0]:.3f} AP70 {ap[4]:.3f} AP85 {ap[7]:.3f} AP90 {ap[8]:.3f}", flush=True)
