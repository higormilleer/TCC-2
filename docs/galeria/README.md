# Galeria da análise qualitativa (E03 × E05)

Esta pasta reúne a galeria da análise qualitativa descrita na Seção 4.7 do TCC (protocolo registrado em
[`configs/qualitativa.yaml`](../../configs/qualitativa.yaml)). Todas as imagens são do conjunto de **validação**, com as
predições da **semente 43** do YOLO26s (E03) e do ASPE residual (E05). O teste não foi usado.

| Arquivo | Conteúdo |
|---|---|
| [`qual_galeria_agropest_1.jpg`](qual_galeria_agropest_1.jpg) | AgroPest-12, uma imagem sorteada por classe, de formiga a tesourinha: referência, E03 e E05 |
| [`qual_galeria_agropest_2.jpg`](qual_galeria_agropest_2.jpg) | AgroPest-12, uma imagem sorteada por classe, de gafanhoto a gorgulho: referência, E03 e E05 |
| [`qual_galeria_ip102.jpg`](qual_galeria_ip102.jpg) | IP102, seis imagens sorteadas da faixa rara: referência, E03 e E05, com a indicação de qual modelo acertou o objeto raro |

As imagens foram sorteadas com semente fixa, e não escolhidas, e as que aparecem nas figuras do Capítulo 5 do TCC foram
excluídas do sorteio. Nas colunas dos modelos aparecem as predições com confiança de pelo menos 0,25. As cores seguem as
figuras da Seção 5.5:

- verde: acerto (classe certa e IoU de pelo menos 0,5);
- laranja: classe trocada;
- azul tracejado: falso positivo;
- cinza tracejado: objeto de referência que o modelo não acertou.

As figuras foram geradas por [`fig_qualitativa.py`](../escrita/figuras/scripts/fig_qualitativa.py), a partir das
predições salvas por `scripts/qualitativa.py`.
