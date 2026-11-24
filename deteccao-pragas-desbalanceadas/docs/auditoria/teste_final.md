# Avaliação final no TESTE (18/11/2026) — única

Lista e regras registradas antes em `configs/teste_final.yaml`; script `scripts/avaliar_teste_final.py`; dados em
`runs/val_eval/teste_final.json`. Checkpoints escolhidos na validação; nada foi mudado depois. fp32.
Fitness = 0,1·mAP50 + 0,9·mAP50-95. Teste: AgroPest-12 1.967 imagens; IP102 3.798 imagens (raras: 18 classes, 75 imagens).

## 1. E05 (ASPE residual) × E03 (YOLO26s), 3 seeds — regras de configs/seed2.yaml
| | Seed 42 | Seed 43 | Seed 44 | Média ± dp |
|---|---|---|---|---|
| AgroPest E03 | 0,4847 | 0,4877 | 0,4859 | 0,4861 ± 0,0015 |
| AgroPest E05 | 0,4927 | 0,4883 | 0,4904 | 0,4904 ± 0,0022 |
| Δ | +0,0079 | +0,0005 | +0,0045 | **+0,0043** |
| IP102 E03 | 0,3806 | 0,3793 | 0,3785 | 0,3794 ± 0,0011 |
| IP102 E05 | 0,3744 | 0,3864 | 0,3869 | 0,3825 ± 0,0071 |
| Δ | −0,0062 | +0,0071 | +0,0084 | +0,0031 |
| IP102 raras AP50-95 Δ | +0,005 | +0,023 | +0,044 | **+0,024** |

- AgroPest: Δ positivo nas 3 seeds, mas média +0,0043 < 0,005. Não atinge "E05 supera E03" no teste (na validação: +0,0112, atingia).
  Ganho concentrado no mAP50-95 (0,4542 → 0,4590); mAP50 igual (0,7735 / 0,7732).
- IP102: sinais mistos → não conclusivo (como na validação; agora a seed negativa é a 42, não a 43).
- IP102 raras: positivo nas 3 seeds e média +0,024 > 0,02 → **"E05 melhora as raras" no teste** (na validação: sinais mistos).
  Comuns: 0,338 → 0,337; frequentes: 0,462 → 0,460 (empate).

## 2. Todos os modelos (seed 42 salvo indicação)
| Modelo | Dataset | Val | Teste |
|---|---|---|---|
| E01 YOLOv8s | AgroPest | 0,4544 | 0,4592 |
| E02 YOLOv8s-ASPE | AgroPest | 0,4547 | 0,4394 |
| E03 YOLO26s | AgroPest | 0,4821 | 0,4847 |
| E03 sem mosaic/mixup (A02) | AgroPest | 0,4657 | 0,4415 |
| E03 conservador (A03) | AgroPest | 0,4866 | 0,4742 |
| E04 YOLO26s-ASPE | AgroPest | 0,4819 | 0,4757 |
| E04 sem augmentation (A05) | AgroPest | 0,3372 | 0,3390 |
| E05 ASPE residual | AgroPest | 0,4985 | 0,4927 |
| E05 sem augmentation (A05) | AgroPest | 0,4207 | 0,3949 |
| E03 YOLO26s | IP102 | 0,3944 | 0,3806 |
| E03 + NorCal γ 0,3 | IP102 | 0,3967 | 0,3794 |
| E04 YOLO26s-ASPE | IP102 | 0,3471 | 0,3410 |
| E04 sem augmentation (A05) | IP102 | 0,1919 | 0,1693 |
| E05 ASPE residual | IP102 | 0,4082 | 0,3744 |
| E05 + NorCal γ 0,2 | IP102 | 0,4123 | 0,3724 |
| E05 + MixUp 0,3 | IP102 | 0,4015 | 0,3943 |
| E05 + RFS | IP102 | 0,3905 | 0,3851 |
| E05 + copy-paste | IP102 | 0,3871 | 0,3821 |
| E05 + RFS + copy-paste | IP102 | 0,3811 | 0,3827 |
| E05 sem augmentation (A05) | IP102 | 0,2518 | 0,2421 |

## 3. Leitura
- ASPE do artigo: pior que o modelo base no teste nos 3 pares (YOLOv8s −0,020; YOLO26s AgroPest −0,009; IP102 −0,040).
- Augmentation: sem augmentation perde 0,10–0,17 no teste (4 casos). A hipótese secundária do A05 (ASPE do artigo perde mais
  que o residual) se confirma no teste nos dois datasets (AgroPest −0,137 × −0,098; IP102 −0,172 × −0,132); na validação só no AgroPest.
  A02 (−0,043) e A03 (−0,011) ficam abaixo do A01 no teste.
- NorCal: sem ganho no teste (−0,001 e −0,002); o ganho da validação vinha de γ escolhido na própria validação.
- Técnicas para raras (seed 42): no teste todas ficam acima do E05 seed 42 (MIX +0,020, RFS +0,011, CP +0,008, RFS+CP +0,008),
  ao contrário da validação. Motivo: o E05 seed 42 foi a melhor seed na validação (0,4082) e a pior no teste (0,3744): checkpoint
  escolhido na validação favorece a validação. Contra a média das 3 seeds do E05 no teste (0,3825): MIX +0,012; RFS +0,003;
  CP −0,000; RFS+CP +0,000. Raras (média E05 0,255): MIX 0,281, RFS+CP 0,277, RFS 0,269, CP 0,265. Só o MixUp se destaca
  no geral, coerente com Crasto (2024); RFS e copy-paste ficam dentro da variação entre seeds (dp 0,0071), seed única.
- Lição metodológica: comparar uma técnica de seed única contra a MESMA seed do modelo base é frágil; a variação entre seeds
  (0,007 no IP102) é do tamanho dos efeitos medidos.
