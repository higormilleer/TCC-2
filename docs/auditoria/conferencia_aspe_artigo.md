# Conferência passo a passo: artigo YOLO-ASPE (Sayni et al., 2025) × nosso código

Feita em 09/11/2026 sobre `2353.pdf` (texto e figuras), Hou et al. (CA, CVPR 2021) e Hu et al. (SE, CVPR 2018).
Código conferido: `src/tcc/models/yolo_aspe.py`.

## 1. Texto do artigo (Passos 1–4) × código

| # | O que o artigo diz | Nosso código | Veredito |
|---|---|---|---|
| Tab. II | Backbone com C2f nos estágios 1–4 (n = 1, 2, 2, 1; canais 64/128/256/512); stem, SPPF, neck e cabeça iguais ao YOLOv8s | `BACKBONE_C2F_LAYERS = (2,4,6,8)`; yolov8s.yaml tem repetições 1,2,2,1 e canais 64/128/256/512; neck/cabeça intactos | **Bate** |
| 1a | U1 = SiLU(BN(Conv1×1(X))), C → C/2 | `cv1 = Conv(c, c//2, 1)` (Conv = conv sem viés + BN + SiLU; o viés b1 é redundante antes do BN) | **Bate** |
| 1b | U2 = SiLU(BN(Conv3×3(U1))), C/2 → C/2 | `cv2 = Conv(h, h, 3, 1)` | **Bate** |
| 2a | F_h = média ao longo de W; F_w = média ao longo de H | `x.mean(dim=3)` e `x.mean(dim=2)` | **Bate** |
| 2b | Concatena F_h e F_w, passa por gargalo compartilhado, divide em G_h e G_w | `cat` + `conv1` + BN + h-swish + `split` | **Bate** (a redução e a ativação não são dadas: escolha nossa) |
| 2c | A_h = σ(W4·G_h), A_w = σ(W5·G_w), ambos com C/2 canais | `conv_h`, `conv_w` + sigmoid | **Bate** |
| 2d | U3 = U2 ⊙ A_h ⊙ A_w | `x * a_h * a_w` | **Bate** |
| 3 | SE sobre U3: z = GAP, s = σ(W2·ReLU(W1·z)), W1 ∈ R^{C/2r × C/2}; U4 = U3 ⊙ s | `SE`: média, fc1 → ReLU → fc2 → sigmoid, `x * s` | **Bate** (r não é dado: escolha nossa) |
| 4 | Y = SiLU(BN(Conv1×1([U4 ; X_id]))), Y ∈ R^{C×H×W} | `cv3 = Conv(h + c, c, 1)` sobre `cat([u, x])` | **Bate**, com X_id = X (C canais), que é leitura nossa |
| Colocação | "C2f-ASPE" = bloco que "integra" o ASPE à estrutura do C2f | C2f padrão **seguido** do ASPE (troca de classe) | **Escolha nossa** (ver seção 3) |

## 2. O que o artigo NÃO especifica (escolhas nossas)
- **CoordAtt:** redução 32, mínimo 8 canais, BatchNorm e h-swish. Vem do código oficial de Hou et al. (o artigo de Hou só diz "ex.: r = 32" e "ativação não linear").
- **SE:** r = 16, mínimo 4 canais, com viés nas FC. Vem de Hu et al. ("r = 16 em todos os experimentos").
- **X_id:** não é definido. Adotamos o próprio X (C canais), o que faz a fusão final ter 1,5·C canais de entrada.
- **Colocação:** depois do C2f, e não dentro dos bottlenecks.
- **Detalhes de conv:** stride 1, padding automático, sem viés antes do BN.
- **Inicialização:** os pesos do C2f vêm do `yolov8s.pt` pré-treinado; os módulos ASPE iniciam aleatórios.

## 3. Onde o próprio artigo se contradiz (não é culpa do código)
1. **Figura 2 × texto.** A Figura 2 ("detailed structure of the C2f-ASPE block") mostra um C2f cujos bottlenecks viram `CIB_CA_Bottleneck`: Conv → Conv → **CA** → **CIB** (max-pool e avg-pool, concat, Conv, sigmoid: atenção espacial) com soma residual. **Não há SE, nem a compressão C → C/2, nem a fusão por concat dos Passos 1–4.** O texto descreve outra coisa.
2. **Figura 1 × texto.** A Figura 1 mostra blocos C2f-ASPE também no **neck** (4 blocos) e só 3 no backbone; o texto diz "estritamente confinado ao backbone", com 4 estágios, e a Tabela VI trata "neck only" e "backbone + neck" como variantes à parte.
3. **Referências cruzadas erradas.** O texto cita "ablation study (Table I)" para a ordem CA→SE, mas a Tabela I é a de contagem de imagens; diz que a Tabela II detalha o neck, mas ela só traz o backbone.
4. **Tabela VI × Coordinate Attention de verdade.** Nosso código mede o custo real: CA no estágio 3 = **6.680** parâmetros; CA nos 4 estágios = 37.368; SE nos 4 estágios = 11.520. O artigo informa +0,4 M (CA só no estágio 3), +1,2 M (CA nos estágios 1–4), +1,0 M (CA "naive") e +0,4 M (SE) — de **30 a 100 vezes** o custo possível dessas atenções. Os parâmetros da Tabela VI não são compatíveis com CA/SE padrão.
5. Na nossa implementação, dos +1,51 M do ASPE, só ~0,05 M são atenção (CA + SE); o resto são as convoluções 1×1 e 3×3. A leitura da Figura 2 (CA + CIB dentro dos bottlenecks) custaria apenas ~0,05 M no total, bem longe dos +1,6 M da Tabela V.

## 4. Tabela V (custo)

| Métrica | Artigo (YOLOv8s → ASPE) | Nosso (YOLOv8s → ASPE) |
|---|---|---|
| Parâmetros | 11,2 M → 12,8 M (+1,6 M, +14%) | 11,14 M → 12,65 M (+1,51 M, +13,5%) |
| GFLOPs | 28,4 → 31,2 (+2,8, +10%) | 28,46 → 32,06 (+3,6, +12,6%) |

O baseline bate (11,14 M ≈ 11,2 M; 28,46 ≈ 28,4), então a contagem é comparável. O ASPE do artigo tem ~0,1 M a mais de parâmetros e ~0,9 GFLOPs a menos que o nosso: **aproxima, mas não reproduz exatamente**. A docstring do código dizia "reproduz"; foi corrigida.

## 5. Conclusão
- **O código segue o texto (Passos 1–4) em todos os pontos verificáveis**; as lacunas do texto foram preenchidas com os valores dos artigos originais de CA e SE.
- **Fidelidade a "o artigo" como um todo não é definível**, porque texto, Figura 1, Figura 2 e Tabela VI descrevem arquiteturas diferentes. Quem implementar só pela Figura 2 obtém outro bloco.
- A probabilidade de nosso bloco ser idêntico ao que o autor treinou é baixa, mas a de **qualquer** implementação independente ser idêntica também é.
- Isso, somado às inconsistências de dataset (`investigacao_dataset_artigo.md`), reforça pedir o código ao autor e levar o ponto ao orientador.
