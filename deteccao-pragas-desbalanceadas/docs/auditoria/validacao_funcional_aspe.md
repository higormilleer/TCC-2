# Validação funcional do ASPE

Ambiente: torch 2.7.1+cu126, dispositivo cuda.

## YOLOv8s-ASPE

- Pesos pré-treinados: `yolov8s.pt`; 349/355 tensores do modelo base transferidos (6 descartados por forma diferente, ex.: camadas de classe do `Detect` com nc=12).
- Tensores novos do ASPE (sem correspondência, inicializados aleatoriamente): 132.
- Camadas convertidas: 4 (índices 2, 4, 6, 8).

| | Parâmetros (M) | GFLOPs @640 |
|---|---|---|
| YOLOv8s original | 11.14 | 28.7 |
| YOLOv8s-ASPE | 12.65 | 32.3 |
| Diferença | +1.51 (+14%) | +3.7 |

| Camada | Tipo | Canais de entrada → saída | Tensores/parâmetros ASPE |
|---|---|---|---|
| 2 | C2fASPE (base C2f) | 64 → 64 | 18812 |
| 4 | C2fASPE (base C2f) | 128 → 128 | 72412 |
| 6 | C2fASPE (base C2f) | 256 → 256 | 285088 |
| 8 | C2fASPE (base C2f) | 512 → 512 | 1131304 |

- Forward (modo treino): saída do tipo `dict`, chaves ['boxes', 'scores', 'feats'].
- Backward: loss = 62.516; **84/84** tensores de parâmetros do ASPE receberam gradiente não nulo.
- Um passo de AdamW alterou **84/84** tensores do ASPE.
- Inferência (modo avaliação): formas de saída [(2, 16, 8400)].
- Salvar e recarregar checkpoint: 4 blocos ASPE recuperados.

## YOLO26s-ASPE

- Pesos pré-treinados: `yolo26s.pt`; 696/708 tensores do modelo base transferidos (12 descartados por forma diferente, ex.: camadas de classe do `Detect` com nc=12).
- Tensores novos do ASPE (sem correspondência, inicializados aleatoriamente): 132.
- Camadas convertidas: 4 (índices 2, 4, 6, 8).

| | Parâmetros (M) | GFLOPs @640 |
|---|---|---|
| YOLO26s original | 9.96 | 22.8 |
| YOLO26s-ASPE | 11.73 | 31.8 |
| Diferença | +1.77 (+18%) | +9.0 |

| Camada | Tipo | Canais de entrada → saída | Tensores/parâmetros ASPE |
|---|---|---|---|
| 2 | C2fASPE (base C3k2) | 64 → 128 | 72412 |
| 4 | C2fASPE (base C3k2) | 128 → 256 | 285088 |
| 6 | C2fASPE (base C3k2) | 256 → 256 | 285088 |
| 8 | C2fASPE (base C3k2) | 512 → 512 | 1131304 |

- Forward (modo treino): saída do tipo `dict`, chaves ['one2many', 'one2one'].
- Backward: loss = 76.526; **84/84** tensores de parâmetros do ASPE receberam gradiente não nulo.
- Um passo de AdamW alterou **84/84** tensores do ASPE.
- Inferência (modo avaliação): formas de saída [(2, 16, 8400)].
- Salvar e recarregar checkpoint: 4 blocos ASPE recuperados.
