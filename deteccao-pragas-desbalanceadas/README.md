# Detecção de Pragas Agrícolas em Cenários Desbalanceados

Código, configurações e resultados do Trabalho de Conclusão de Curso **"Detecção de Pragas Agrícolas em Cenários
Desbalanceados"**, apresentado ao curso de Bacharelado em Ciência da Computação da Universidade Tecnológica Federal do
Paraná (UTFPR), Câmpus Santa Helena, em 2026.

- **Autor:** Higor Miller Grassi
- **Orientador:** Prof. Dr. Anderson Brilhador

O trabalho avaliou estratégias de arquitetura e de dados para detectores YOLO em bases de pragas com classes
desbalanceadas e propôs o **ASPE residual**, uma forma de ligar o módulo de atenção ASPE (Sayni et al., 2025) a um
detector pré-treinado sem prejudicar o que ele já aprendeu.

## Sumário

- [O que foi feito](#o-que-foi-feito)
- [ASPE residual](#aspe-residual)
- [Principais resultados](#principais-resultados)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Ambiente](#ambiente)
- [Dados](#dados)
- [Como reproduzir os experimentos](#como-reproduzir-os-experimentos)
- [Avaliação e análises](#avaliação-e-análises)
- [Registro dos experimentos](#registro-dos-experimentos)
- [Como citar](#como-citar)
- [Licença](#licença)

## O que foi feito

- **Detectores:** YOLOv8s e YOLO26s, com e sem o módulo ASPE, a partir dos pesos pré-treinados no COCO.
- **Bases:** AgroPest-12 (12 classes, quase balanceada, usada como controle) e IP102 (102 classes; razão de 974:1 entre a
  classe mais e a menos frequente no treino).
- **Divisões sem vazamento:** as duas bases tinham cópias e quase duplicatas da mesma foto em subconjuntos diferentes. As
  imagens foram agrupadas por *hash* perceptual de diferença (dHash) antes da divisão, e as listas finais estão em
  [`splits/`](splits).
- **Técnicas avaliadas:** ablação do aumento de dados, reamostragem por fator de repetição (RFS), Copy-Paste, MixUp
  reforçado, recalibração das pontuações (NorCal) e, de forma complementar, perda ponderada e Focal Loss.
- **Protocolo:** hipóteses e regras de decisão registradas antes dos experimentos, decisões tomadas só na validação,
  comparações centrais com três sementes (42, 43 e 44) e conjunto de teste usado **uma única vez**, em 18/11/2026.

| Código | Modelo |
|---|---|
| E01 | YOLOv8s |
| E02 | YOLOv8s com o ASPE original (em série) |
| E03 | YOLO26s |
| E04 | YOLO26s com o ASPE original (em série) |
| E05 | YOLO26s com o ASPE residual (proposto) |

## ASPE residual

No ASPE original, a saída de cada bloco do *backbone* é substituída pela saída do módulo, que começa com pesos aleatórios.
No ASPE residual, o módulo vira um ramo somado à saída do bloco original *B*:

$$Y = B(X) + \mathrm{ASPE}(B(X))$$

A última normalização em lote do ramo começa com γ = β = 0. Assim, no início do ajuste fino, o ramo produz zero e a rede
é idêntica à pré-treinada; o módulo passa a contribuir aos poucos, à medida que aprende. A implementação está em
[`src/tcc/models/yolo_aspe.py`](src/tcc/models/yolo_aspe.py).

## Principais resultados

*Fitness* = 0,1 · mAP@0,5 + 0,9 · mAP@0,5:0,95. Média ± desvio-padrão de três sementes.

| Base | Conjunto | YOLO26s (E03) | ASPE residual (E05) | Diferença |
|---|---|---|---|---|
| AgroPest-12 | Validação | 0,4883 ± 0,0054 | **0,4995 ± 0,0048** | +0,0112 |
| AgroPest-12 | Teste | 0,4861 ± 0,0015 | **0,4904 ± 0,0022** | +0,0043 |
| IP102 | Validação | 0,3941 ± 0,0032 | **0,3978 ± 0,0099** | +0,0037 |
| IP102 | Teste | 0,3794 ± 0,0011 | **0,3825 ± 0,0071** | +0,0031 |

- O ASPE original, reimplementado a partir da descrição do artigo, **não reproduziu o ganho relatado** em nenhum dos dois
  detectores.
- O ASPE residual eliminou essa perda e superou o YOLO26s nas três sementes na validação do AgroPest-12. No teste, o ganho
  médio ficou abaixo do limiar de decisão registrado (0,005). No IP102, o resultado geral foi inconclusivo; na faixa
  rara, o AP@0,5:0,95 do teste passou de 0,231 para 0,255.
- Uma ablação mostrou que a soma residual e a inicialização em zero contribuíram em partes semelhantes.
- O **aumento de dados** foi o fator de maior efeito: retirá-lo reduziu a *fitness* entre 0,08 e 0,16 na validação.
- As técnicas para classes raras não trouxeram ganhos consistentes; o MixUp reforçado foi a exceção no teste, como
  indício obtido com uma semente.

Comparação das arquiteturas no AgroPest-12 (semente 42; mAP no teste):

| Modelo | mAP@0,5 | mAP@0,5:0,95 | *Fitness* teste | *Fitness* validação | Parâmetros (M) | Latência (ms) |
|---|---|---|---|---|---|---|
| E01 YOLOv8s | 0,7439 | 0,4275 | 0,4592 | 0,4544 | 11,14 | 3,7 |
| E02 YOLOv8s-ASPE | 0,6988 | 0,4106 | 0,4394 | 0,4547 | 12,65 | 5,8 |
| E03 YOLO26s | 0,7770 | 0,4523 | 0,4847 | 0,4821 | 9,96 | 5,8 |
| E04 YOLO26s-ASPE | 0,7445 | 0,4458 | 0,4757 | 0,4819 | 11,73 | 8,1 |
| E05 ASPE residual | 0,7763 | 0,4611 | 0,4927 | 0,4985 | 11,73 | 8,1 |

Latência mediana por imagem, com NMS, lote de uma imagem, precisão simples e entrada de 640 × 640 pixels, em uma NVIDIA
GeForce RTX 4060 Ti. Os números completos estão em [`runs/val_eval/`](runs/val_eval) e no texto do TCC.

## Estrutura do repositório

```text
.
├── configs/                  # protocolo e registros dos experimentos (com a data de cada registro)
├── docs/
│   ├── auditoria/            # auditorias dos dados, conferência do ASPE e registros de decisões
│   └── escrita/figuras/scripts/  # scripts que geraram as figuras do TCC
├── runs/
│   ├── val_eval/             # métricas de validação e do teste único (JSON)
│   ├── historico_treinos.csv # métricas por época de cada treino
│   └── registro_treinos.csv  # configuração, pasta e duração de cada treino
├── scripts/                  # preparação dos dados, treino, avaliação e análises
├── splits/                   # listas das divisões de treino, validação e teste usadas no TCC
└── src/tcc/                  # pacote com a conversão das bases, o ASPE, as perdas e as técnicas para classes raras
```

## Ambiente

Os experimentos foram feitos no Windows 11, com uma NVIDIA GeForce RTX 4060 Ti de 8 GB, Python 3.9.13, PyTorch 2.7.1
(CUDA 12.6) e Ultralytics 8.4.165. Para recriar o ambiente, instale primeiro o PyTorch com suporte a CUDA e, depois, as
demais dependências:

```powershell
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Os comandos abaixo devem ser executados na raiz do repositório. Os pesos pré-treinados `yolov8s.pt` e `yolo26s.pt` ficam
na raiz; quando não estão lá, a Ultralytics os baixa automaticamente no primeiro treino.

## Dados

As bases **não são redistribuídas** neste repositório. Baixe-as das fontes oficiais:

- **AgroPest-12:** <https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset> (licença MIT).
- **IP102** (subconjunto de detecção): <https://github.com/xpwu95/IP102> (uso acadêmico, com a citação do artigo original).

1. **Baixar e converter para o formato YOLO.** O download do AgroPest-12 usa o Kaggle CLI (`pip install kaggle` e o
   arquivo `kaggle.json` com sua chave), e o do IP102 usa o `gdown`. Se preferir baixar manualmente, coloque os arquivos
   em `data/raw/agropest12/` e `data/raw/ip102/`.

   ```powershell
   python scripts/preparar_dados.py baixar
   python scripts/preparar_dados.py converter
   ```

2. **Montar as divisões usadas no TCC.** As listas em `splits/` reproduzem exatamente as divisões dos experimentos, e o
   script confere o resultado com os manifestos gravados quando as divisões foram criadas.

   ```powershell
   python scripts/aplicar_divisoes.py agropest12
   python scripts/aplicar_divisoes.py ip102
   ```

   Para regenerar as divisões do zero, com o agrupamento por dHash, use `python scripts/make_split_plano.py` (AgroPest-12)
   e `python scripts/limpar_vazamento_ip102.py` (IP102).

| Base | Treino | Validação | Teste |
|---|---|---|---|
| AgroPest-12 | 9.209 | 1.967 | 1.967 |
| IP102 | 11.924 | 2.182 | 3.798 |

No AgroPest-12, as listas trazem o nome dos arquivos de imagem; no IP102, o identificador de cada imagem (nome sem
extensão). As auditorias que justificam as divisões estão em [`docs/auditoria/`](docs/auditoria).

## Como reproduzir os experimentos

Todos os treinos usam o protocolo de [`configs/protocolo_plano.yaml`](configs/protocolo_plano.yaml): AdamW, taxa de
aprendizado inicial de 0,001, lote de 16 imagens, entrada de 640 × 640 pixels, até 300 épocas com paciência de 50 e
escolha do *checkpoint* pela *fitness* de validação. Cada treino retoma sozinho do último *checkpoint* se for
interrompido.

```powershell
# arquiteturas no AgroPest-12 (semente 42)
python scripts/treinar_plano.py E01
python scripts/treinar_plano.py E02
python scripts/treinar_plano.py E03
python scripts/treinar_plano.py E04
python scripts/treinar_plano.py E04 --aspe-res                       # E05, ASPE residual

# os mesmos modelos no IP102
python scripts/treinar_plano.py E03 --dataset ip102_limpo
python scripts/treinar_plano.py E04 --aspe-res --dataset ip102_limpo

# sementes 43 e 44 do E03 e do E05 nas duas bases (treina, avalia na validação e gera o relatório)
python scripts/rodar_seeds.py

# ablação do aumento de dados (A02: sem Mosaic e MixUp; A03: conservador; A05: sem aumento)
python scripts/treinar_plano.py E03 --aug A02
python scripts/treinar_plano.py E03 --aug A03
python scripts/treinar_plano.py E04 --aug A05
python scripts/treinar_plano.py E04 --aug A05 --aspe-res

# técnicas para as classes raras do IP102 (MixUp reforçado, RFS, Copy-Paste e RFS + Copy-Paste) e NorCal
python scripts/rodar_ip102_minoritarias.py

# análises complementares, feitas depois do teste e avaliadas só na validação
python scripts/treinar_plano.py E04 --aspe-res --sem-zero            # ablação da inicialização em zero
python scripts/treinar_plano.py E04 --aspe-res --perda W             # perda ponderada
python scripts/treinar_plano.py E04 --aspe-res --perda FL            # Focal Loss
python scripts/treinar_plano.py E02 --aspe-fig2                      # leitura alternativa do ASPE (Figura 2 do artigo)

# estudo da taxa de aprendizado inicial (Apêndice A do TCC)
python scripts/treinar_plano.py E01 --lr0 0.002 --epochs 60 --tag lr0_002
```

Acrescente `--dataset ip102_limpo` para treinar qualquer variante no IP102 e `--seed N` para outra semente. Os nomes
dos treinos seguem o padrão `plano_E04_s42_ares` (AgroPest-12) ou `ip102_limpo_E04_s42_ares` (IP102), e a configuração
de cada treino feito no TCC está em [`runs/registro_treinos.csv`](runs/registro_treinos.csv).

## Avaliação e análises

```powershell
python scripts/avaliar_plano.py val plano_E03_s42 plano_E04_s42_ares   # métricas na validação
python scripts/eficiencia_plano.py                                       # parâmetros, GFLOPs e latência
python scripts/mostrar_resultados.py                                     # resume os resultados no terminal
python scripts/qualitativa.py preds                                      # análise qualitativa E03 × E05 (validação)
python scripts/qualitativa.py analise
```

O script [`scripts/avaliar_teste_final.py`](scripts/avaliar_teste_final.py) avalia, uma única vez, os modelos listados em
[`configs/teste_final.yaml`](configs/teste_final.yaml). Ele se recusa a rodar quando o arquivo
`runs/val_eval/teste_final.json` já existe, como neste repositório, que traz o resultado do teste feito em 18/11/2026.
Os demais scripts de [`scripts/`](scripts) contêm, no início, uma descrição do que fazem e de como usá-los, e os scripts
das figuras estão em [`docs/escrita/figuras/scripts/`](docs/escrita/figuras/scripts).

## Registro dos experimentos

Cada grupo de experimentos foi registrado antes de seus resultados, com a pergunta, a hipótese, os parâmetros e a regra
de decisão. A exceção é o registro de 10/11/2026, que documentou uma regra de desempate definida depois de conhecidos os
resultados do primeiro grupo.

| Data | Arquivo | Grupo |
|---|---|---|
| 03/11/2026 | [`protocolo_plano.yaml`](configs/protocolo_plano.yaml) | Protocolo comum e arquiteturas (E01 a E04) |
| 04/11/2026 | [`estudo_lr.yaml`](configs/estudo_lr.yaml) | Taxa de aprendizado inicial |
| 04/11/2026 | [`ablacao_plano.yaml`](configs/ablacao_plano.yaml) | Ablação do aumento de dados (A02 e A03; A05 em 16/11/2026) |
| 10/11/2026 | [`selecao_candidato.md`](docs/auditoria/selecao_candidato.md) | Escolha do modelo de partida |
| 12/11/2026 | [`minoritarias_ip102_limpo.yaml`](configs/minoritarias_ip102_limpo.yaml) | Técnicas para as classes raras do IP102 |
| 12/11/2026 | [`minoritarias.yaml`](configs/minoritarias.yaml) | Técnicas para o AgroPest-12 (só uma época foi executada, sem resultado avaliado) |
| 14/11/2026 | [`seed2.yaml`](configs/seed2.yaml) | Repetição do E03 e do E05 com as sementes 43 e 44 |
| 18/11/2026 | [`teste_final.yaml`](configs/teste_final.yaml) | Avaliação única no teste |
| 18/11/2026 | [`aspe_fig2.yaml`](configs/aspe_fig2.yaml), [`perda_classes.yaml`](configs/perda_classes.yaml) e [`aspe_res_sem_zero.yaml`](configs/aspe_res_sem_zero.yaml) | Análises posteriores ao teste, só na validação |
| 23/11/2026 | [`qualitativa.yaml`](configs/qualitativa.yaml) | Análise qualitativa E03 × E05, só na validação |

## Como citar

GRASSI, Higor Miller. **Detecção de pragas agrícolas em cenários desbalanceados**. 2026. Trabalho de Conclusão de Curso
(Bacharelado em Ciência da Computação) – Universidade Tecnológica Federal do Paraná, Santa Helena, 2026.

```bibtex
@misc{grassi2026pragas,
  author = {Grassi, Higor Miller},
  title  = {Detecção de Pragas Agrícolas em Cenários Desbalanceados},
  year   = {2026},
  note   = {Trabalho de Conclusão de Curso (Bacharelado em Ciência da Computação),
            Universidade Tecnológica Federal do Paraná, Santa Helena}
}
```

## Licença

O código deste repositório está sob a licença [GNU AGPL-3.0](LICENSE), a mesma da biblioteca Ultralytics, da qual ele
depende e cujas classes estende. O texto do TCC está sob a licença Creative Commons Atribuição 4.0 Internacional. As
bases de dados seguem as licenças de seus autores e não fazem parte deste repositório.
