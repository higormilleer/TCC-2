"""Modificações na função de perda de classificação (pós-teste; configs/perda_classes.yaml).

* W  (weighted loss): usa o mecanismo nativo do Ultralytics 8.4 (`cls_pw`): a BCE de cada classe é multiplicada por
  w_c = (1/n_c)^cls_pw, normalizado para média 1 (DetectionTrainer.set_class_weights). Não precisa de código aqui.
* FL (Focal Loss, Lin et al., ICCV 2017): troca a BCE de classificação (elemento a elemento) por
  FL = BCE * (1 - p_t)^gamma * [alpha*y + (1-alpha)*(1-y)], com p_t = y*p + (1-y)*(1-p). Os alvos y do YOLO são
  "suaves" (TaskAlignedAssigner, entre 0 e 1); a fórmula acima vale para alvos suaves (mesma forma da FocalLoss do
  próprio Ultralytics, mas sem reduzir, pois o v8DetectionLoss soma e normaliza depois).

A troca é feita na CLASSE do modelo (init_criterion), e não no objeto da perda, porque o Ultralytics recria a perda
ao retomar o treino (YOLO26: init_criterion() em resume_training) e a remove dos checkpoints.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class FocalBCE(nn.Module):
    """Focal Loss sobre logits, sem redução (substitui nn.BCEWithLogitsLoss(reduction='none'))."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = float(gamma), float(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        p = pred.sigmoid()
        p_t = label * p + (1 - label) * (1 - p)
        loss = loss * (1.0 - p_t) ** self.gamma
        if self.alpha >= 0:
            loss = loss * (label * self.alpha + (1 - label) * (1 - self.alpha))
        return loss


class FocalDetectionModel(DetectionModel):
    """DetectionModel cuja perda de classificação é a Focal Loss (vale para YOLOv8 e para o E2ELoss do YOLO26)."""

    focal_gamma = 2.0
    focal_alpha = 0.25

    def init_criterion(self):
        crit = super().init_criterion()
        partes = [crit] if isinstance(crit, v8DetectionLoss) else [crit.one2many, crit.one2one]
        for p in partes:
            assert isinstance(p, v8DetectionLoss), type(p)
            p.bce = FocalBCE(self.focal_gamma, self.focal_alpha)
        return crit


def trainer_focal(base, gamma: float, alpha: float):
    """Subclasse do trainer (ex.: ASPEResTrainer) cujo modelo usa a Focal Loss na classificação."""

    class FocalTrainer(base):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            assert type(model) is DetectionModel, f"modelo inesperado: {type(model)}"
            model.__class__ = FocalDetectionModel
            model.focal_gamma, model.focal_alpha = float(gamma), float(alpha)
            return model

    FocalTrainer.__name__ = FocalTrainer.__qualname__ = f"Focal{base.__name__}"
    return FocalTrainer
