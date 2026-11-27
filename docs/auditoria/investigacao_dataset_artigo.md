# Investigação: por que o AgroPest-12 do artigo YOLO-ASPE difere do nosso

Investigado em 04/11/2026. Todas as evidências abaixo foram verificadas nos arquivos e nas páginas citadas.

## 1. A nossa cópia está íntegra
- `agropest.zip` (baixado em 28/09/2026): 13.143 imagens e 13.143 rótulos (`train` 11.502, `valid` 1.095, `test` 546).
- Os 13.143 rótulos processados no projeto são **idênticos, byte a byte**, aos do zip original.
- Nenhuma imagem ilegível; nenhum rótulo malformado.

## 2. Não existe outra versão pública
Página oficial (kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset), lida em 04/11/2026:
- Autor Rupankar Majumdar; licença MIT; **"Expected update frequency: Never"**; "Updated a year ago".
- A descrição **não informa** contagem de imagens, contagem por classe, origem Flickr, resolução de 300 px, drones ou cajueiros.
- O `data.yaml` da página é igual ao nosso (mesmos 12 nomes, na mesma ordem).
- Citação pedida pelo autor: *"AgroPest-12: A 12-Class Image Dataset of Crop Insects and Pests," Kaggle, 2025* (o título difere do usado em alguns textos).

## 3. O que o artigo afirma e o dado público mostra

| Afirmação do artigo (YOLO-ASPE) | Dado público (nossa cópia) |
|---|---|
| Tabela I: imagens por classe (ex.: Weevils 819, "menor classe"; Caterpillars 1.312) | Por prefixo do nome do arquivo: Weevils 1.122, Caterpillars 1.041; **menor classe = Earthworms (797)**. Por presença da classe nos rótulos o resultado é praticamente o mesmo. Nenhum critério reproduz a Tabela I. |
| Imagens do Flickr, "redimensionadas para no máximo 300 px" | **Todas as imagens (amostra de 1.500) têm 640 × 640** px. |
| Pragas minúsculas ("10 × 10 px", "menos de 0,5% da imagem") | Mediana da caixa ≈ **348 px** (raiz da área, em 640 px). **92,4% das caixas > 96 px (grandes), 7,3% médias, 0,4% < 32 px (pequenas).** |
| Contexto: cajueiros, drones a 5–15 m, África Ocidental | O dataset descreve "ambientes agrícolas naturais"; sem menção a cajueiro ou drone. |
| Soma por classe = 13.143 imagens (uma classe por imagem) | Há 28 imagens com mais de uma classe e 67 cujo rótulo difere da classe do nome do arquivo. |
| Divisão 70/15/15 = 9.200 / 1.971 / 1.972 | Reprodutível (obtivemos 9.202 / 1.970 / 1.971), mas o sorteio por imagem vaza cópias aumentadas da mesma foto entre treino e teste (2.479 de 5.488 fotos). |

## 4. Conclusão
A divergência **não vem do nosso processamento nem de versão do dataset**. A Tabela I e a descrição do conjunto no artigo não são reproduzíveis com a versão pública. As explicações possíveis são: tabela aproximada/incorreta ou uso de outra versão dos dados (o artigo diz que estão "com o autor correspondente"). Não é possível decidir entre elas sem contato com os autores.

## 5. Consequência metodológica importante
O artigo justifica o ASPE por detectar **objetos pequenos**. No AgroPest-12 público, **apenas 0,4% das caixas são pequenas** (< 32 px). Logo:
- o mAP para objetos pequenos (mAP_S) quase não tem exemplos neste dataset;
- a premissa "pragas minúsculas" não se aplica aos dados que usamos;
- isso é compatível com o resultado observado (o ASPE não superou o baseline), embora não prove a causa.
Recomendação: relatar a distribuição de tamanhos de objeto na caracterização do dataset e evitar afirmar que o trabalho avalia detecção de pragas pequenas.
