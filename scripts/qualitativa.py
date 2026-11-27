#!/usr/bin/env python
"""Análise qualitativa E03 x E05 (protocolo: configs/qualitativa.yaml) — SÓ VALIDAÇÃO.

Uso:
  python scripts/qualitativa.py preds      # salva as predições do val (CPU) em runs/val_eval/pred/<run>/predictions.json
  python scripts/qualitativa.py analise    # categorias a-d, IoU, confusões, faixa rara -> runs/val_eval/qualitativa.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TAU = 0.25
CFG = yaml.safe_load(open(ROOT / "configs" / "qualitativa.yaml", encoding="utf-8"))
PARES = {  # base -> [(semente, E03, E05)]
    "agropest12": [(43, "plano_E03_s43", "plano_E04_s43_ares"), (44, "plano_E03_s44", "plano_E04_s44_ares")],
    "ip102": [(43, "ip102_limpo_E03_s43", "ip102_limpo_E04_s43_ares"), (44, "ip102_limpo_E03_s44", "ip102_limpo_E04_s44_ares")],
}
OUT = ROOT / "runs" / "val_eval"


def info(run):
    return yaml.safe_load(open(ROOT / "runs" / "logs" / f"plano_{run}.yaml", encoding="utf-8"))


def preds():
    import torch
    from tcc.models.yolo_aspe import load_fp32
    torch.set_num_threads(8)  # deixa CPU livre para o carregamento de dados do treino em andamento
    for base in PARES.values():
        for _, *runs in base:
            for run in runs:
                dest = OUT / "pred" / run / "predictions.json"
                if dest.exists():
                    print("já existe:", run, flush=True)
                    continue
                inf = info(run)
                y, _ = load_fp32(Path(inf["best"]).with_name("best_fp32.pt"), device="cpu")
                m = y.val(data=inf["data"], split="val", half=False, plots=False, workers=0, verbose=False, save_json=True,
                          batch=8, device="cpu", project=str(OUT / "pred"), name=run, exist_ok=True)
                f = 0.1 * m.box.map50 + 0.9 * m.box.map
                print(f"{run}: mAP50 {m.box.map50:.4f} mAP50-95 {m.box.map:.4f} fitness {f:.4f}", flush=True)


# ------------------------------------------------------------------ análise (mesmos critérios de diagnostico_aspe.py)
def load_gt(data_yaml):
    ds = Path(yaml.safe_load(open(data_yaml, encoding="utf-8"))["path"])
    gt = {}
    for lf in sorted((ds / "labels" / "val").glob("*.txt")):
        im = next(iter((ds / "images" / "val").glob(lf.stem + ".*")))
        W, H = Image.open(im).size
        rows = []
        for l in lf.read_text().splitlines():
            p = l.split()
            if len(p) >= 5:
                c, x, y, w, h = int(float(p[0])), *map(float, p[1:5])
                rows.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
        gt[lf.stem] = (np.array(rows, dtype=float).reshape(-1, 5), str(im))
    return gt


def load_pred(run):
    J = json.loads((OUT / "pred" / run / "predictions.json").read_text())
    off = 0 if min(j["category_id"] for j in J) == 0 else 1
    P = {}
    for j in J:
        if j["score"] < TAU:
            continue
        x, y, w, h = j["bbox"]
        P.setdefault(str(j["image_id"]), []).append((j["category_id"] - off, x, y, x + w, y + h, j["score"]))
    return {k: np.array(v) for k, v in P.items()}


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def objetos(gt, P):
    """Uma linha por objeto de referência: imagem, classe, categoria, IoU com a melhor predição da mesma classe,
    classe da predição trocada (se houver); e as predições por imagem com seu status (para as figuras)."""
    obj, fps, status_pred = [], 0, {}
    for stem, (g, _) in gt.items():
        p = P.get(stem, np.zeros((0, 6)))
        M = iou_mat(p[:, 1:5], g[:, 1:5])
        same = (p[:, 0][:, None] == g[:, 0][None, :]) if len(p) and len(g) else np.zeros((len(p), len(g)), bool)
        Ms = np.where(same, M, 0.0)
        for j in range(len(g)):
            bs = Ms[:, j].max() if len(p) else 0.0
            outra = (M[:, j] * (p[:, 0] != g[j, 0])) if len(p) else np.zeros(0)
            oc = int(p[outra.argmax(), 0]) if len(p) and outra.max() >= .5 else -1
            cat = "ok" if bs >= .5 else ("cls" if oc >= 0 else ("loc" if bs >= .1 else "miss"))
            obj.append((stem, j, int(g[j, 0]), cat, float(bs), oc))
        st = []
        for i in range(len(p)):
            if len(g) and Ms[i].max() >= .5:
                st.append("ok")
            elif len(g) and (M[i] * (g[:, 0] != p[i, 0])).max() >= .5:
                st.append("cls")
            else:
                st.append("fp")
                fps += 1
        status_pred[stem] = (p, st)
    return obj, fps, status_pred


def analise():
    mf = yaml.safe_load(open(ROOT / "configs" / "minoritarias_ip102_limpo.yaml", encoding="utf-8"))["faixas_de_frequencia"]
    raras = set(mf["raras_menos_de_20"])
    res, det = {}, {}
    for base, pares in PARES.items():
        gt = load_gt(info(pares[0][1])["data"])
        for sem, r3, r5 in pares:
            o3, fp3, s3 = objetos(gt, load_pred(r3))
            o5, fp5, s5 = objetos(gt, load_pred(r5))
            assert [x[:3] for x in o3] == [x[:3] for x in o5]
            ok3 = np.array([x[3] == "ok" for x in o3]); ok5 = np.array([x[3] == "ok" for x in o5])
            iou3 = np.array([x[4] for x in o3]); iou5 = np.array([x[4] for x in o5])
            cls_ = np.array([x[2] for x in o3])
            c = ok3 & ok5
            d_iou = iou5[c] - iou3[c]
            r = {
                "n_obj": int(len(o3)), "a": int((ok5 & ~ok3).sum()), "b": int((ok3 & ~ok5).sum()),
                "c": int(c.sum()), "d": int((~ok3 & ~ok5).sum()),
                "acertos_E03": int(ok3.sum()), "acertos_E05": int(ok5.sum()),
                "classe_trocada_E03": int(sum(x[3] == "cls" for x in o3)), "classe_trocada_E05": int(sum(x[3] == "cls" for x in o5)),
                "fp_E03": fp3, "fp_E05": fp5,
                "iou_c_mediana_E03": float(np.median(iou3[c])), "iou_c_mediana_E05": float(np.median(iou5[c])),
                "iou_c_dif_mediana": float(np.median(d_iou)), "iou_c_frac_E05_maior": float((d_iou > 0).mean()),
                "iou_c_frac_maior_que_0.01": float((d_iou > 0.01).mean()), "iou_c_frac_menor_que_-0.01": float((d_iou < -0.01).mean()),
                "frac_iou85_E03": float((iou3[c] >= .85).mean()), "frac_iou85_E05": float((iou5[c] >= .85).mean()),
            }
            # IC de 95% (bootstrap por imagem, 2000 réplicas) da diferença de mediana de IoU nos objetos c
            stems = np.array([x[0] for x in o3])[c]
            u, inv = np.unique(stems, return_inverse=True)
            rng = np.random.default_rng(0)
            meds = []
            for _ in range(2000):
                pick = rng.integers(0, len(u), len(u))
                w = np.bincount(pick, minlength=len(u))[inv]
                dd = np.repeat(d_iou, w)
                meds.append(np.median(dd) if len(dd) else 0)
            r["iou_c_dif_mediana_ic95"] = [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]
            if base == "ip102":
                rr = np.isin(cls_, list(raras))
                r.update({"raras_n_obj": int(rr.sum()), "raras_acertos_E03": int((ok3 & rr).sum()),
                          "raras_acertos_E05": int((ok5 & rr).sum()), "raras_a": int((ok5 & ~ok3 & rr).sum()),
                          "raras_b": int((ok3 & ~ok5 & rr).sum()),
                          "raras_imagens_val": int(len({x[0] for x, m in zip(o3, rr) if m}))})
            res[f"{base}_s{sem}"] = r
            det[f"{base}_s{sem}"] = {"o3": o3, "o5": o5, "s3": s3, "s5": s5, "gt": gt}
            print(base, sem, json.dumps(r, ensure_ascii=False), flush=True)
    # confusões (classe trocada) no AgroPest-12, sementes somadas
    conf = {}
    for k in ("agropest12_s43", "agropest12_s44"):
        for mod in ("o3", "o5"):
            for x in det[k][mod]:
                if x[3] == "cls":
                    conf.setdefault((x[2], x[5]), {"o3": 0, "o5": 0})[mod] += 1
    pares = sorted(conf.items(), key=lambda kv: -abs(kv[1]["o5"] - kv[1]["o3"]))[:8]
    res["agropest12_confusoes_que_mais_mudaram"] = [{"verdadeira": a, "predita": b, "E03": v["o3"], "E05": v["o5"]} for (a, b), v in pares]
    print(json.dumps(res["agropest12_confusoes_que_mais_mudaram"], ensure_ascii=False))
    json.dump(res, open(OUT / "qualitativa.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pickle.dump(det, open(OUT / "qualitativa_det.pkl", "wb"))


if __name__ == "__main__":
    {"preds": preds, "analise": analise}[sys.argv[1]]()
