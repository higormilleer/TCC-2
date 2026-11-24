"""YOLO-ASPE — reimplementação de Sayni et al. (2025), Int. J. Adv. Res. 13(12):291-304.

YOLOv8s com os 4 blocos C2f do backbone trocados por blocos C2f-ASPE (Axis-Separated
Positional Encoding). Neck e cabeça ficam idênticos ao YOLOv8s (como diz o artigo).

O artigo NÃO publica código e deixa ambíguos alguns detalhes. Interpretação adotada
(documentada em docs/METHOD_ADAPTATIONS.md):

* O bloco ASPE (Passos 1–4 do artigo) opera em X com C canais e devolve C canais:
    U1 = SiLU(BN(Conv1x1(X)))            C   -> C/2
    U2 = SiLU(BN(Conv3x3(U1)))           C/2 -> C/2
    Ah, Aw = Coordinate Attention(U2)    pooling 1-D por eixo, gargalo compartilhado, sigmoid
    U3 = U2 * Ah * Aw
    U4 = U3 * SE(U3)                     recalibração de canais, sequencial CA -> SE
    Y  = SiLU(BN(Conv1x1([U4 ; X_id])))  X_id = identidade (X), saída com C canais
* "C2f-ASPE" = C2f padrão (com os mesmos n repetições) seguido do bloco ASPE sobre a
  saída do C2f. Essa leitura APROXIMA o custo informado na Tabela V do artigo (artigo: +1,6 M de
  parâmetros e +2,8 GFLOPs; aqui: +1,51 M e +3,6 GFLOPs); trocar o C2f inteiro pelo bloco ASPE REDUZIRIA
  os parâmetros, em vez de aumentá-los. ATENÇÃO: a Figura 2 do artigo mostra outra estrutura (CA + CIB dentro
  de cada bottleneck, sem SE), que contradiz o texto (Passos 1-4); seguimos o texto.
  Ver docs/auditoria/conferencia_aspe_artigo.md.
* Reduções não informadas pelo artigo: CA usa redução 32 (mínimo 8 canais), como no
  Coordinate Attention original; SE usa r = 16.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2f, Conv
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.utils.torch_utils import unwrap_model

BACKBONE_C2F_LAYERS = (2, 4, 6, 8)  # estágios 1–4 (Tabela II do artigo); no YOLO26s as camadas 2,4,6,8 são C3k2 (subclasse de C2f)


class CoordAtt(nn.Module):
    """Coordinate Attention (Hou et al., 2021): descritores separados por linha e coluna."""

    def __init__(self, c: int, reduction: int = 32):
        super().__init__()
        mip = max(8, c // reduction)
        self.conv1 = nn.Conv2d(c, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, c, 1)
        self.conv_w = nn.Conv2d(mip, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        f_h = x.mean(dim=3, keepdim=True)  # (N,C,H,1)  — média ao longo de W
        f_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (N,C,W,1) — média ao longo de H
        g = self.act(self.bn1(self.conv1(torch.cat([f_h, f_w], dim=2))))
        g_h, g_w = torch.split(g, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(g_h))  # (N,C,H,1)
        a_w = torch.sigmoid(self.conv_w(g_w.permute(0, 1, 3, 2)))  # (N,C,1,W)
        return x * a_h * a_w


class SE(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018)."""

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        mid = max(4, c // r)
        self.fc1 = nn.Conv2d(c, mid, 1)
        self.fc2 = nn.Conv2d(mid, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * s


class ASPEUnit(nn.Module):
    def __init__(self, c: int, zero_init: bool = False):
        super().__init__()
        h = c // 2
        self.cv1 = Conv(c, h, 1, 1)  # U1
        self.cv2 = Conv(h, h, 3, 1)  # U2
        self.ca = CoordAtt(h)
        self.se = SE(h)
        self.cv3 = Conv(h + c, c, 1, 1)  # fusão com a identidade
        if zero_init:  # variante E05 (NÃO é do artigo): saída do bloco = 0 no início, para a soma residual preservar o C2f pré-treinado
            nn.init.zeros_(self.cv3.bn.weight)
            nn.init.zeros_(self.cv3.bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.cv2(self.cv1(x))
        u = self.se(self.ca(u))
        return self.cv3(torch.cat([u, x], dim=1))


class C2fASPE(C2f):
    """C2f padrão + bloco ASPE sobre a saída. Criado por troca de classe de um C2f existente."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aspe(super().forward(x))


class C2fASPERes(C2fASPE):
    """Variante E05 (deslocamento DECLARADO em relação ao artigo): y = C2f(x); saída = y + ASPE(y), com o ASPE
    iniciado em zero. No início da otimização o modelo é idêntico ao pré-treinado; o ASPE só soma correções."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = C2f.forward(self, x)
        return y + self.aspe(y)


def convert_backbone(model, residual: bool = False, zero_init: bool | None = None) -> int:
    """Troca os C2f dos estágios 1–4 do backbone por C2f-ASPE, preservando os pesos do C2f.
    residual=True cria a variante E05 (soma residual + inicialização em zero).
    zero_init=None segue `residual`; residual=True com zero_init=False é a ablação "_nz" (configs/aspe_res_sem_zero.yaml)."""
    zi = residual if zero_init is None else bool(zero_init)
    n = 0
    for i in BACKBONE_C2F_LAYERS:
        m = model.model[i]
        if isinstance(m, C2fASPE):
            continue
        assert isinstance(m, C2f), f"camada {i} não é C2f/C3k2: {type(m)}"
        c2 = m.cv2.conv.out_channels
        m.__class__ = C2fASPERes if residual else C2fASPE
        m.aspe = ASPEUnit(c2, zero_init=zi)
        n += 1
    return n


class SpatialAtt(nn.Module):
    """Atenção espacial (o "CIB" da Figura 2 do artigo): média e máximo ao longo dos canais, concatenação, conv e sigmoid.
    O artigo não informa o kernel; usa-se 7×7, como no módulo espacial do CBAM (Woo et al., ECCV 2018)."""

    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(a))


class BottleneckCACIB(Bottleneck):
    """Bottleneck da Figura 2 do artigo ("CIB_CA_Bottleneck"): Conv -> Conv -> CA -> CIB, com a soma residual do bottleneck.
    Variante "ASPE-Fig2" (18/11/2026, posterior ao teste; ver configs/aspe_fig2.yaml): testa se a leitura pela figura, e não pelo texto,
    reproduz o ganho relatado. Os pesos cv1/cv2 vêm do pré-treino; CA e CIB começam aleatórios."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cib(self.ca(self.cv2(self.cv1(x))))
        return x + y if self.add else y


def convert_backbone_fig2(model) -> int:
    """Troca os bottlenecks dos C2f dos estágios 1–4 do backbone por BottleneckCACIB (leitura da Figura 2 do artigo)."""
    n = 0
    for i in BACKBONE_C2F_LAYERS:
        m = model.model[i]
        assert type(m) is C2f, f"camada {i}: a variante Fig2 é só para o YOLOv8s (C2f), achou {type(m)}"
        for b in m.m:
            if isinstance(b, BottleneckCACIB):
                continue
            c = b.cv2.conv.out_channels
            b.__class__ = BottleneckCACIB
            b.ca = CoordAtt(c)
            b.cib = SpatialAtt()
            n += 1
    return n


class FP32Trainer(DetectionTrainer):
    """Trainer que, além do best.pt/last.pt (fp16) do Ultralytics, grava best_fp32.pt/last_fp32.pt.

    Motivo: com AdamW e lr0=0,01 (valores do artigo) as variâncias de BatchNorm passam de 65504,
    o limite do fp16. O Ultralytics grava o EMA em fp16 e aplica nan_to_num, que satura esses valores;
    o checkpoint recarregado fica inutilizável (mAP de 0,09 contra 0,91 medido durante o treino).
    O EMA vivo está em fp32 e está correto, então guardamos o state_dict dele em fp32.
    """

    def resume_training(self, ckpt):
        """O Ultralytics 8.4.165 NÃO restaura o estado do EarlyStopping ao retomar: a paciência passa a contar a partir da
        primeira época após a retomada (no E05 o treino foi até a época 158 com melhor época 61). Aqui o estado é refeito
        a partir do results.csv, para a paciência de 50 valer como num treino sem interrupção."""
        super().resume_training(ckpt)
        f = self.save_dir / "results.csv"
        if ckpt is None or not self.resume or not f.exists():
            return
        import csv

        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if not rows:
            return
        k50 = next(k for k in rows[0] if "mAP50(B)" in k)
        k95 = next(k for k in rows[0] if "mAP50-95(B)" in k)
        fit, ep = max((0.1 * float(r[k50]) + 0.9 * float(r[k95]), int(float(r["epoch"]))) for r in rows)
        self.stopper.best_fitness, self.stopper.best_epoch = fit, ep
        from ultralytics.utils import LOGGER

        LOGGER.info(f"EarlyStopping restaurado do results.csv: melhor época {ep} (fitness {fit:.4f})")

    def save_model(self):
        super().save_model()
        ema = unwrap_model(self.ema.ema)
        sd = {
            k: (v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu())
            for k, v in ema.state_dict().items()
        }
        payload = {
            "epoch": self.epoch,
            "fitness": None if self.fitness is None else float(self.fitness),
            "state_dict": sd,
            "names": dict(ema.names),
            "aspe": any(isinstance(m, C2fASPE) for m in ema.model),
            "aspe_res": any(isinstance(m, C2fASPERes) for m in ema.model),
            "aspe_fig2": any(isinstance(m, BottleneckCACIB) for m in ema.modules()),
            "aspe_zero_init": getattr(self, "ZERO_INIT", None),  # só registro (None = padrão: zero se residual)
            "yaml": dict(ema.yaml),  # arquitetura (YOLOv8s ou YOLO26s) para recarregar sem depender do nome
        }
        torch.save(payload, self.wdir / "last_fp32.pt")
        if self.best_fitness == self.fitness:
            torch.save(payload, self.wdir / "best_fp32.pt")


def _carregar_depois(trainer_get_model, converter, cfg, weights, verbose):
    """Cria o modelo, converte os blocos e SÓ ENTÃO carrega os pesos.

    Correção de 18/11/2026: antes os pesos eram carregados no modelo ainda sem ASPE e a conversão vinha depois. No treino
    novo isso não muda nada (o yolo*.pt não tem ASPE), mas na RETOMADA (resume) os pesos treinados do ASPE eram
    descartados (intersect_dicts) e o ASPE do modelo vivo voltava ao estado inicial; só o EMA guardava o ASPE treinado.
    Rodadas afetadas (retomadas antes da melhor época): plano_E04_s42_ares (ép. 23), plano_E02_s42_afig2 (ép. 14),
    ip102_limpo_E04_s43_ares_repro (ép. 50), ip102_limpo_E04_s42_ares (ép. 3)."""
    model = trainer_get_model(cfg, None, verbose)
    converter(model)
    if weights:
        model.load(weights)
    return model


class ASPETrainer(FP32Trainer):
    """Trainer do Ultralytics que aplica a troca dos blocos antes de criar o otimizador."""

    RESIDUAL = False
    ZERO_INIT = None  # None = padrão (zero se residual)

    def get_model(self, cfg=None, weights=None, verbose=True):
        conv = lambda m: convert_backbone(m, residual=self.RESIDUAL, zero_init=self.ZERO_INIT)  # noqa: E731
        return _carregar_depois(super().get_model, conv, cfg, weights, verbose)


class ASPEResTrainer(ASPETrainer):
    """Variante E05: ASPE residual com inicialização em zero (ver C2fASPERes)."""

    RESIDUAL = True


class ASPEResNZTrainer(ASPEResTrainer):
    """Ablação "_nz" (configs/aspe_res_sem_zero.yaml): soma residual do E05, mas ASPE com inicialização PADRÃO (sem zero).
    Separa os dois ingredientes do E05: a soma y + ASPE(y) e o início em zero."""

    ZERO_INIT = False


class ASPEFig2Trainer(FP32Trainer):
    """Variante ASPE-Fig2: CA + CIB dentro dos bottlenecks do backbone (leitura da Figura 2 do artigo)."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        return _carregar_depois(super().get_model, convert_backbone_fig2, cfg, weights, verbose)


def load_fp32(path, device="cpu"):
    """Reconstrói o modelo (YOLOv8s ou YOLO-ASPE) a partir de um *_fp32.pt e devolve um YOLO pronto para val()."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    nc = len(ck["names"])
    model = DetectionModel(ck.get("yaml") or "yolov8s.yaml", nc=nc, verbose=False)
    if ck["aspe"]:
        convert_backbone(model, residual=bool(ck.get("aspe_res", False)))
    if ck.get("aspe_fig2"):
        convert_backbone_fig2(model)
    model.load_state_dict(ck["state_dict"])
    model.names = ck["names"]
    y = YOLO("yolov8s.yaml")  # apenas o invólucro; o modelo real (qualquer arquitetura) é atribuído abaixo
    y.model = model.to(device).float().eval()
    return y, ck
