"""Augmentations focadas nas classes minoritárias (extensão do E05 = YOLO26s-ASPE residual). Ver configs/minoritarias.yaml.

Três mecanismos independentes, que podem ser combinados:

* RFS  (sobreamostragem): imagens que contêm alguma classe-alvo são sorteadas com peso `rfs_peso` (as demais com peso 1).
        O número de imagens por época NÃO muda (amostragem com reposição, num_samples = tamanho do treino), então o custo
        de computação é o mesmo do E05. Como o Mosaic sorteia os outros 3 ladrilhos do buffer de imagens recentes, o viés
        também chega aos ladrilhos.
* CP   (copy-paste das classes-alvo): com probabilidade `cp_p` por imagem carregada (cada ladrilho do Mosaic é uma imagem),
        cola 1 instância de uma classe-alvo (classe sorteada uniformemente, depois a instância) recortada pela caixa de outra
        imagem do treino. Sem máscaras de segmentação no AgroPest-12: o recorte é retangular, com borda suavizada.
* CPC  (copy-paste de pares confundidos): com probabilidade `cpc_p`, se a imagem contém uma classe de um grupo de espécies
        que o E05 confunde na validação, cola uma instância de OUTRA classe do mesmo grupo (ex.: Slug numa foto de Earthworm).
        Objetivo: obrigar o modelo a separar espécies parecidas lado a lado.

A colagem acontece ANTES do Mosaic e do resto do pipeline do Ultralytics (que segue igual ao protocolo A01), apenas no
conjunto de TREINO. Validação e teste não são tocados.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
import torch
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances

MAX_LADO_RECORTE = 320  # px, na escala da imagem já redimensionada para imgsz (640)
MIN_LADO_RECORTE = 12  # caixas menores que isso não entram no banco
AREA_MAX_COLAGEM = 0.25  # fração da área da imagem
OCLUSAO_MAX = 0.30  # fração máxima de uma caixa existente (ou da colada) coberta pela outra


class MinorityDataset(YOLODataset):
    """YOLODataset de treino com copy-paste por classe. Criado por troca de classe (ds.__class__) após o build normal."""

    def configurar(self, mm: dict) -> None:
        self.mm = dict(mm)
        self.mm_alvo = [int(c) for c in mm.get("classes_alvo", [])]
        self.mm_parceiros = {int(k): [int(x) for x in v] for k, v in (mm.get("parceiros") or {}).items()}
        precisa = set(self.mm_alvo) if mm.get("cp_p", 0) > 0 else set()
        if mm.get("cpc_p", 0) > 0:
            for v in self.mm_parceiros.values():
                precisa.update(v)
        self.mm_banco = self._montar_banco(sorted(precisa))

    def _montar_banco(self, classes: list[int]) -> dict[int, list[tuple[bytes, float]]]:
        """Recorta todas as instâncias das classes pedidas (na escala de treino) e guarda em JPEG na memória."""
        banco = {c: [] for c in classes}
        if not classes:
            return banco
        for lb in self.labels:
            cls = lb["cls"].reshape(-1).astype(int)
            sel = np.isin(cls, classes)
            if not sel.any():
                continue
            im = cv2.imread(lb["im_file"])
            if im is None:
                continue
            h0, w0 = im.shape[:2]
            r = self.imgsz / max(h0, w0)
            if r != 1:
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=cv2.INTER_LINEAR)
            h, w = im.shape[:2]
            for c, (x, y, bw, bh) in zip(cls[sel], lb["bboxes"][sel]):  # xywh normalizado
                x1, y1 = int(round((x - bw / 2) * w)), int(round((y - bh / 2) * h))
                x2, y2 = int(round((x + bw / 2) * w)), int(round((y + bh / 2) * h))
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if min(x2 - x1, y2 - y1) < MIN_LADO_RECORTE:
                    continue
                crop = im[y1:y2, x1:x2]
                s = MAX_LADO_RECORTE / max(crop.shape[:2])
                if s < 1:
                    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                                      interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    banco[int(c)].append((buf.tobytes(), float(bw * bh)))
        LOGGER.info("copy-paste: banco de recortes " + ", ".join(f"classe {c}: {len(v)}" for c, v in banco.items()))
        return banco

    def get_image_and_label(self, index: int) -> dict:
        label = super().get_image_and_label(index)
        if not self.augment:
            return label
        colar = []
        if self.mm.get("cp_p", 0) > 0 and self.mm_alvo and random.random() < self.mm["cp_p"]:
            colar.append(random.choice(self.mm_alvo))
        if self.mm.get("cpc_p", 0) > 0 and self.mm_parceiros:
            presentes = set(label["cls"].reshape(-1).astype(int).tolist())
            opcoes = sorted({p for c in presentes for p in self.mm_parceiros.get(c, [])})
            if opcoes and random.random() < self.mm["cpc_p"]:
                colar.append(random.choice(opcoes))
        colar = [c for c in colar if self.mm_banco.get(c)]
        return self._colar(label, colar) if colar else label

    def _colar(self, label: dict, classes: list[int]) -> dict:
        img = label["img"].copy()  # load_image devolve o array do cache/buffer: nunca alterar no lugar
        h, w = img.shape[:2]
        inst = label["instances"]
        inst.convert_bbox("xyxy")
        inst.denormalize(w, h)
        boxes = [b for b in inst.bboxes.astype(np.float32)]
        cls = label["cls"].reshape(-1).astype(np.float32).tolist()
        for c in classes:
            buf, area_rel = random.choice(self.mm_banco[c])
            crop = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            ch, cw = crop.shape[:2]
            e0, e1 = self.mm.get("cp_escala", (0.5, 1.0))
            alvo = min(area_rel * random.uniform(e0, e1), self.mm.get("cp_area_max", AREA_MAX_COLAGEM)) * h * w  # tamanho natural da espécie
            s = math.sqrt(alvo / (ch * cw))
            nw, nh = max(MIN_LADO_RECORTE, round(cw * s)), max(MIN_LADO_RECORTE, round(ch * s))
            if nw >= w or nh >= h:
                continue
            crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if random.random() < 0.5:
                crop = crop[:, ::-1]
            pos = None
            for _ in range(15):
                x1, y1 = random.randint(0, w - nw), random.randint(0, h - nh)
                b = np.array([x1, y1, x1 + nw, y1 + nh], dtype=np.float32)
                if all(_ok_sobreposicao(b, e) for e in boxes):
                    pos = b
                    break
            if pos is None:
                continue
            x1, y1 = int(pos[0]), int(pos[1])
            a = _mascara_suave(nh, nw)[..., None]
            reg = img[y1:y1 + nh, x1:x1 + nw].astype(np.float32)
            img[y1:y1 + nh, x1:x1 + nw] = (a * crop.astype(np.float32) + (1 - a) * reg).round().astype(img.dtype)
            boxes.append(pos)
            cls.append(float(c))
        bb = np.stack(boxes).astype(np.float32) if boxes else np.zeros((0, 4), np.float32)
        novo = Instances(bb, inst.segments, None, bbox_format="xyxy", normalized=False)
        novo.convert_bbox("xywh")
        novo.normalize(w, h)
        label["img"] = img
        label["instances"] = novo
        label["cls"] = np.array(cls, dtype=np.float32).reshape(-1, 1)
        return label


def _ok_sobreposicao(b: np.ndarray, e: np.ndarray) -> bool:
    iw = max(0.0, min(b[2], e[2]) - max(b[0], e[0]))
    ih = max(0.0, min(b[3], e[3]) - max(b[1], e[1]))
    inter = iw * ih
    ae = max(1e-6, (e[2] - e[0]) * (e[3] - e[1]))
    ab = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / ae < OCLUSAO_MAX and inter / ab < OCLUSAO_MAX


def _mascara_suave(h: int, w: int) -> np.ndarray:
    """1 no centro, rampa linear até ~0 numa borda de 10% do menor lado (reduz a costura do recorte retangular)."""
    m = max(2, int(0.1 * min(h, w)))
    ry = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1) / m
    rx = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1) / m
    return np.clip(np.minimum(ry[:, None], rx[None, :]), 0, 1).astype(np.float32)


def rfs_ativo(mm: dict) -> bool:
    return mm.get("rfs_t", 0) > 0 or mm.get("rfs_peso", 1.0) > 1.0


def pesos_rfs(dataset, mm: dict) -> torch.Tensor:
    """Peso de amostragem de cada imagem de treino.
    rfs_t > 0  -> Repeat Factor Sampling do LVIS (Gupta et al., 2019): r_c = max(1, sqrt(t / f_c)), f_c = fração das imagens
                  de treino que contêm a classe c; peso da imagem = maior r_c entre as classes presentes.
    rfs_peso   -> peso fixo para imagens com alguma classe de 'classes_alvo' (versão usada no AgroPest-12)."""
    cls_img = [set(lb["cls"].reshape(-1).astype(int).tolist()) for lb in dataset.labels]
    if mm.get("rfs_t", 0) > 0:
        n = len(cls_img)
        cont = {}
        for cs in cls_img:
            for c in cs:
                cont[c] = cont.get(c, 0) + 1
        r = {c: max(1.0, math.sqrt(mm["rfs_t"] / (k / n))) for c, k in cont.items()}
        return torch.tensor([max((r[c] for c in cs), default=1.0) for cs in cls_img], dtype=torch.double)
    alvo = set(int(c) for c in mm["classes_alvo"])
    return torch.tensor([mm["rfs_peso"] if alvo & cs else 1.0 for cs in cls_img], dtype=torch.double)


VARIANTES = ("RFS", "CP", "CPC", "MIX")  # ordem canônica (nome do run e combinação)
AUG_ULTRALYTICS = ("mixup",)  # parâmetros que vão direto para o Ultralytics (variante MIX), não para o trainer


def config_combinada(cfg_yaml: dict, variantes: list[str]) -> dict:
    """Une os parâmetros das variantes pedidas (ex.: ['RFS', 'CP']) num único dicionário para o trainer."""
    mm = {"classes_alvo": list(cfg_yaml["classes_alvo"]), "parceiros": dict(cfg_yaml.get("parceiros") or {}),
          "rfs_peso": 1.0, "rfs_t": 0.0, "cp_p": 0.0, "cpc_p": 0.0}
    mm.update(cfg_yaml.get("geometria_colagem") or {})
    for v in variantes:
        if v not in cfg_yaml["variantes"]:
            raise ValueError(f"variante {v} não existe neste config (disponíveis: {list(cfg_yaml['variantes'])})")
        mm.update(cfg_yaml["variantes"][v]["parametros"])
    return mm


def trainer_minoritarias(base, mm: dict):
    """Cria uma subclasse do trainer (ex.: ASPEResTrainer) que aplica RFS / CP / CPC só no loader de TREINO."""

    class MinorityTrainer(base):
        MM = dict(mm)

        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == "train" and (self.MM["cp_p"] > 0 or self.MM["cpc_p"] > 0):
                assert type(ds) is YOLODataset, f"dataset inesperado: {type(ds)}"
                ds.__class__ = MinorityDataset
                ds.configurar(self.MM)
            return ds

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train" or not rfs_ativo(self.MM):
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            ds = self.build_dataset(dataset_path, mode, batch_size)
            p = pesos_rfs(ds, self.MM)
            LOGGER.info(f"RFS: {int((p > 1).sum())}/{len(p)} imagens de treino com peso > 1 (máx {float(p.max()):.2f}, "
                        f"fração das amostras nessas imagens {float(p[p > 1].sum() / p.sum()):.2f})")
            return loader_ponderado(ds, batch_size, self.args.workers, p)

    MinorityTrainer.__name__ = MinorityTrainer.__qualname__ = f"Minority{base.__name__}"
    return MinorityTrainer


def loader_ponderado(dataset, batch: int, workers: int, pesos: torch.Tensor) -> InfiniteDataLoader:
    """Igual ao build_dataloader do Ultralytics (GPU única), trocando o embaralhamento uniforme por amostragem ponderada."""
    import os

    seed = torch.initial_seed() - 1
    g_sampler = torch.Generator()
    g_sampler.manual_seed(seed % (1 << 63))
    sampler = torch.utils.data.WeightedRandomSampler(pesos, num_samples=len(dataset), replacement=True, generator=g_sampler)
    nb = math.ceil(len(dataset) / batch)
    nw = min(os.cpu_count() // max(torch.cuda.device_count(), 1), workers, 0 if nb <= 1 else nb)
    generator = torch.Generator()
    generator.manual_seed((6148914691236517205 + seed) % (1 << 64))
    return InfiniteDataLoader(
        dataset=dataset, batch_size=batch, shuffle=False, num_workers=nw, sampler=sampler,
        prefetch_factor=4 if nw > 0 else None, pin_memory=torch.cuda.is_available(),
        collate_fn=getattr(dataset, "collate_fn", None), worker_init_fn=seed_worker, generator=generator, drop_last=False,
    )
