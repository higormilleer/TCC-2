#!/usr/bin/env python
"""Validação da limpeza do IP102 (item 13 do parecer): efeito por faixa e checagem dos pares do dHash.

1. Instâncias e imagens por faixa de frequência (rara/comum/frequente) em treino, validação e teste, antes (ip102_v2)
   e depois (ip102_limpo) da remoção de quase duplicatas.
2. Pares com Hamming <= 3 entre subconjuntos diferentes: distribuição da distância e correlação de Pearson das imagens
   em 64x64 tons de cinza (medida independente do dHash), para estimar falsos positivos.
3. Folha de inspeção visual com 40 pares sorteados (numpy default_rng(2026)).
Saídas: docs/auditoria/ip102_dhash_validacao.json e docs/escrita/figuras/dhash_amostra.jpg
"""
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "ip102_v2"
antes = json.loads((ROOT / "data" / "splits" / "ip102_v2" / "manifest.json").read_text(encoding="utf-8"))
depois = json.loads((ROOT / "docs" / "auditoria" / "ip102_limpo_manifesto.json").read_text(encoding="utf-8"))
F = yaml.safe_load(open(ROOT / "configs" / "minoritarias_ip102_limpo.yaml", encoding="utf-8"))["faixas_de_frequencia"]
faixa = {}
for k, v in F.items():
    nomef = "rara" if k.startswith("raras") else ("comum" if k.startswith("comuns") else ("frequente" if k.startswith("freq") else None))
    if nomef and isinstance(v, list):
        for c in v:
            faixa[int(c)] = nomef
img = {p.stem: p for p in (SRC / "images_all").iterdir()}


def classes(stem):
    f = SRC / "labels_all" / f"{stem}.txt"
    return [int(float(l.split()[0])) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def contagem(man, chaves):
    out = {}
    for nome_sp, k in chaves.items():
        inst = {"rara": 0, "comum": 0, "frequente": 0, "fora_das_faixas": 0}
        ims = {"rara": set(), "comum": set(), "frequente": set()}
        for s in man[k]:
            for c in classes(s):
                fx = faixa.get(c, "fora_das_faixas")
                inst[fx] += 1
                if fx in ims:
                    ims[fx].add(s)
        out[nome_sp] = {"instancias": inst, "imagens": {k2: len(v) for k2, v in ims.items()}, "n_imagens": len(man[k])}
    return out


res = {"faixas_definidas_no_treino_limpo": {k: sum(1 for v in faixa.values() if v == k) for k in ("rara", "comum", "frequente")},
       "antes": contagem(antes, {"treino": "train", "validacao": "val", "teste": "holdout"}),
       "depois": contagem(depois, {"treino": "train", "validacao": "val", "teste": "test"}),
       "removidos": depois["removidos"]}

# ---- pares do dHash entre subconjuntos (mesma função do scripts/limpar_vazamento_ip102.py)
split_of = {s: sp for sp in ("train", "val", "holdout") for s in antes[sp]}


def dhash(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    s = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    return int(np.packbits((s[:, 1:] > s[:, :-1]).flatten()).view(">u8")[0])


stems = sorted(s for s in split_of if s in img)
H = np.array([dhash(img[s]) for s in stems], dtype=np.uint64)


def popcount(a):
    a = a - ((a >> np.uint64(1)) & np.uint64(0x5555555555555555))
    a = (a & np.uint64(0x3333333333333333)) + ((a >> np.uint64(2)) & np.uint64(0x3333333333333333))
    a = (a + (a >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((a * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(int)


pares = []
for i in range(0, len(H), 400):
    d = popcount(H[i:i + 400, None] ^ H[None, :])
    ii, jj = np.nonzero(d <= 3)
    for a, b in zip(ii + i, jj):
        if a < b and split_of[stems[a]] != split_of[stems[b]]:
            pares.append((stems[a], stems[b], int(d[a - i, b])))


def pequena(s):
    im = cv2.imread(str(img[s]), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


cors = []
for a, b, _ in pares:
    x, y = pequena(a), pequena(b)
    cors.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 1.0)
cors = np.array(cors)
dist = np.array([p[2] for p in pares])
res["pares_entre_subconjuntos"] = {
    "n": len(pares), "por_distancia": {str(k): int((dist == k).sum()) for k in range(4)},
    "correlacao_64px": {"mediana": float(np.median(cors)), "p05": float(np.percentile(cors, 5)), "min": float(cors.min()),
                        "frac_maior_0_9": float((cors > 0.9).mean()), "frac_maior_0_8": float((cors > 0.8).mean()),
                        "n_menor_0_8": int((cors <= 0.8).sum())},
    "tipos": {k: int(sum(1 for a, b, _ in pares if {split_of[a], split_of[b]} == set(k.split("-")))) for k in ("train-holdout", "val-holdout", "train-val")},
}

# ---- folha de inspeção: 40 pares sorteados
rng = np.random.default_rng(2026)
sel = rng.choice(len(pares), size=min(40, len(pares)), replace=False)
cel = 150
folha = np.full((10 * (cel + 26), 4 * (2 * cel + 16), 3), 255, np.uint8)
for n, k in enumerate(sel):
    a, b, d = pares[k]
    r, c = divmod(n, 4)
    y0, x0 = r * (cel + 26), c * (2 * cel + 16)
    for j, s in enumerate((a, b)):
        im = cv2.resize(cv2.imread(str(img[s])), (cel, cel), interpolation=cv2.INTER_AREA)
        folha[y0:y0 + cel, x0 + j * cel:x0 + (j + 1) * cel] = im
    cv2.putText(folha, f"{n + 1}: d={d} r={cors[k]:.2f}", (x0 + 4, y0 + cel + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.imwrite(str(ROOT / "docs" / "escrita" / "figuras" / "dhash_amostra.jpg"), folha, [cv2.IMWRITE_JPEG_QUALITY, 88])
res["amostra_inspecao"] = [{"n": i + 1, "a": pares[k][0], "b": pares[k][1], "split_a": split_of[pares[k][0]], "split_b": split_of[pares[k][1]],
                            "hamming": pares[k][2], "correlacao": round(float(cors[k]), 3)} for i, k in enumerate(sel)]
baixos = np.argsort(cors)[:12]
res["pares_menor_correlacao"] = [{"a": pares[k][0], "b": pares[k][1], "hamming": pares[k][2], "correlacao": round(float(cors[k]), 3)} for k in baixos]
(ROOT / "docs" / "auditoria" / "ip102_dhash_validacao.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("amostra_inspecao", "pares_menor_correlacao")}, ensure_ascii=False, indent=1))
print("menor correlação:", res["pares_menor_correlacao"][:6])
