# IP102 para o experimento (11/11/2026): dados regenerados e sem vazamento

## O que foi feito
1. **Mapeamento de classes corrigido** (`src/tcc/data/prepare.py`, duas ocorrências do bug: 0-based, sem subtrair 1). Regenerado em `data/processed/ip102_v2` (a pasta antiga `ip102` ficou intacta; o Windows não permitiu renomear). Verificação (`scripts/verificar_ip102_v2.py`): **18.975 imagens com id de classe igual ao do XML, 0 inconsistências**.
2. **Vazamento encontrado** no split oficial (treino/val/teste) por dHash (Hamming ≤ 3): 424 imagens de val (16%) e 613 de teste (16%) com quase-duplicata no treino; 194 do teste com quase-duplicata na val.
3. **Limpeza** (`scripts/limpar_vazamento_ip102.py`): união de imagens quase idênticas em grupos (4.088 pares, 16.667 grupos, maior grupo 29 imagens). O **teste oficial fica intacto**; os membros de treino/val de grupos que tocam o teste foram removidos; grupos mistos treino↔val vão para o treino. Resultado em `data/processed/ip102_limpo`, conferido de novo: **0 quase-duplicatas entre splits**.

| Split | v2 (oficial) | limpo |
|---|---|---|
| treino | 12.498 | 11.924 (−884 removidas) |
| val | 2.679 | 2.182 (−187 removidas) |
| teste | 3.798 | 3.798 (intacto) |

## Distribuição (limpo)
- Treino: 13.895 instâncias, 97 classes com instâncias, mín. 2, máx. 1.949 (razão **974:1**). Val: 95 classes. Teste: 96 classes.
- Classes sem nenhuma instância em nenhum split: 59, 60, 63, 75, 80 (5 classes do IP102 não têm anotação de detecção). O mAP é calculado sobre as classes presentes na val.

## Protocolo de treino
Igual ao do plano (`configs/protocolo_plano.yaml`: lr0 0,001, batch 16, AdamW, 300 épocas, patience 50, mosaic 1,0 com close_mosaic 50, seed 42), com `nc=102`. O lr0 foi escolhido no AgroPest-12 e **não** foi reestudado no IP102 (limitação a declarar).
Comandos: `python scripts/treinar_plano.py E03 --dataset ip102_limpo` e `... E04 --dataset ip102_limpo`.
Tempo medido em 1 época de teste: E03 ≈ 0,046 h, E04 ≈ 0,070 h (inclui carga e validação final).
