# IP102: erro no mapeamento de classes da conversão XML -> YOLO (achado em 04/11/2026)

## O que acontecia
O `<name>` de cada objeto nos XMLs do IP102 é o índice de classe **0-based** (valores de 0 a 101; há 190 caixas com 0 e
nenhuma com 102). O conversor tratava valores de 1 a 102 como 1-based e subtraía 1. Resultado, medido comparando o XML
bruto com os rótulos processados (18.974 imagens):

| XML | id no nosso rótulo |
|---|---|
| 0 (rice leaf roller) | 0 |
| 1 (rice leaf caterpillar) | **0** (fundida com a classe 0) |
| 2..101 | 1..100 (deslocadas em -1) |
| — | 101 nunca usado |

## Efeitos
- Treino e métricas globais (mAP) continuam internamente consistentes: o detector aprende os ids que recebeu.
- Duas espécies distintas (190 + 130 caixas) viram uma só classe.
- **Todos os nomes de classe dos relatórios do IP102 estão deslocados em um** (o id k aparece com o nome names[k], mas contém a classe verdadeira k+1).
- A lista de "classes sem exemplo de treino" deve ser refeita: no XML bruto 5 classes não têm nenhuma caixa
  (Viteus vitifoliae, Colomerus vitis, Polyphagotars onemus latus, Phyllocoptes oleiverus ashmead, Parlatoria zizyphus Lucus).
- Os grupos minoritária/majoritária e a estratificação do validation foram calculados sobre ids deslocados e com uma classe fundida.

## Situação
Conversor corrigido em `src/tcc/data/prepare.py`. Os dados processados do IP102 e todos os experimentos IP102 anteriores
(E0-E10, seeds 42-44) devem ser tratados como **exploratórios** até a regeração. O AgroPest-12 não é afetado
(o mapeamento dele vem do `data.yaml`; rótulos idênticos ao zip original).
