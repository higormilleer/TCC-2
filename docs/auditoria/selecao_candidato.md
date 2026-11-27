# Seleção do modelo candidato (etapa 6 do plano) — 10/11/2026

Decisão tomada **somente com a validação** (fp32, melhor checkpoint por fitness). O conjunto de teste não foi usado.
Orientação do professor (10/11/2026, por mensagem): seguir o planejamento, escolher o melhor modelo e passar para a próxima etapa, sem usar os resultados do artigo como referência.

## Resultados na validação (E01–E04, seed 42, uma execução cada)
| | Modelo | mAP50 | mAP50-95 | Fitness | Params (M) | GFLOPs | Latência completa (ms) | FPS |
|---|---|---|---|---|---|---|---|---|
| E01 | YOLOv8s | 0,7328 | 0,4235 | 0,4544 | 11,14 | 28,5 | 4,1 | 241 |
| E02 | YOLOv8s-ASPE | 0,7068 | 0,4267 | 0,4547 | 12,65 | 32,1 | 6,0 | 166 |
| **E03** | **YOLO26s** | **0,7606** | 0,4511 | **0,4821** | **9,96** | **20,8** | 6,1 | 165 |
| E04 | YOLO26s-ASPE | 0,7396 | **0,4532** | 0,4819 | 11,73 | 29,7 | 8,3 | 121 |

(Latência: batch 1, fp32, 640, RTX 4060 Ti com ~30% de uso de fundo; E03 e E04 em modo NMS externo.)

## Regra aplicada
1. **Critério principal:** maior fitness (0,1·mAP50 + 0,9·mAP50-95) na validação.
2. **Limiar de ruído (registrado antes dos treinos):** diferenças < 0,005 são inconclusivas (desvio entre seeds 0,001–0,003; 1 seed por modelo).
3. **Desempate (definido nesta data, depois de ver E01–E04):** entre modelos empatados dentro do ruído, vale o de menor custo computacional (parâmetros, GFLOPs, memória).
   A regra de desempate foi escrita depois dos resultados de E01–E04; por isso é declarada aqui como tal. Ela não usa o teste.

## Decisão: **E03 (YOLO26s, sem ASPE)**
- E03 × E01: +0,0277 de fitness, muito acima do ruído → a arquitetura YOLO26s é melhor que o YOLOv8s neste cenário, com menos parâmetros e GFLOPs.
- E04 × E03: −0,0002, empate dentro do ruído → o ASPE não trouxe ganho mensurável no YOLO26s.
- Em empate, E03 custa menos: −1,8 M de parâmetros, −8,9 GFLOPs (−30%), ~26% mais rápido (165 contra 121 FPS).
- E02 × E01: +0,0003, empate; E01 é mais rápido (241 contra 166 FPS).

## Ressalvas (devem constar no TCC)
- Uma seed por modelo. O empate E03 × E04 não prova que o ASPE seja inútil; prova que, aqui, o ganho (se existe) é menor que o ruído medido.
- O ASPE teve mAP50 menor (−2,1 e −2,6 pontos) e mAP50-95 um pouco maior (+0,2 e +0,3 ponto) nos dois pares.
- Early stopping encerrou os treinos antes da época 251, então a fase sem Mosaic (`close_mosaic` 50) nunca foi atingida em E01–E04.
- A comparação com o artigo não é reprodução direta (split, lr, seleção de checkpoint e descrição do dataset diferem; ver `investigacao_dataset_artigo.md` e `conferencia_aspe_artigo.md`).

## Próxima etapa: ablação de augmentation (Seção 8) sobre E03
- **A01:** referência = o próprio treino E03 (não repetir). Nome: `plano_E03_s42`.
- **A02:** sem Mosaic e MixUp. Comando: `python scripts/treinar_plano.py E03 --aug A02`. Run: `plano_E03_s42_A02`.
- **A03:** augmentation conservador (valores de `configs/ablacao_plano.yaml`, aprovados pelo orientador na instrução de 10/11/2026 de seguir o plano). Comando: `python scripts/treinar_plano.py E03 --aug A03`. Run: `plano_E03_s42_A03`.
- Tudo o mais é idêntico ao protocolo congelado (split, lr0 0,001, batch, épocas, seed, critério de checkpoint). Decisão da ablação só pela validação.
