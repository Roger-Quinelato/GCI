# Dados e análise da competição GCI

Os arquivos originais do Drive ficam em `input/`. `train.csv` contém `TARGET`; `test.csv` não contém o alvo. O arquivo `sample_submission.csv` define a ordem dos IDs e o formato de entrega.

## Ler os resultados

Comece por `notebooks/base/analise_exploratoria.ipynb`: é o notebook base para acompanhar célula por célula. A sequência é importar → describe → info → head → entender as colunas → outliers → correlação → exploração complementar. Somente depois começa o tratamento, seguido da preparação de uma divisão comum para modelos futuros. As tabelas e os gráficos aparecem dentro do notebook.

O relatório anterior está em `analise/relatorio_analise.html`. A versão anterior do notebook foi preservada em `notebooks/referencia/analise_exploratoria_detalhada.ipynb` como aprofundamento. Esses resultados anteriores documentam a antiga divisão 70/30. A divisão atual 80/20 e seus tamanhos estão no notebook base e em `dados_modelagem/preparacao.json`. As tabelas completas daquela análise estão em `analise/*.csv`.

As duas formas dos dados têm finalidades diferentes:

- `dados_tratados/train.csv` e `dados_tratados/test.csv`: limpeza de sentinelas, marcadores e domínios inválidos. Preservam valores ausentes e as mesmas colunas dos originais. Use para explorar os dados e refazer a preparação em novas partições.
- `dados_modelagem/treino.csv`, `validacao.csv` e `teste.csv`: partições numéricas, com medianas e one-hot ajustados somente no subtreino. A divisão estratificada do notebook base usa 80% para treino e 20% para validação, com semente 42. O teste fornecido pelo curso permanece separado. Os arquivos auxiliares registram medianas, colunas e partições.

Antes de treinar o ensemble com os dados de modelagem:

```python
import pandas as pd

treino = pd.read_csv("dados_modelagem/treino.csv")
validacao = pd.read_csv("dados_modelagem/validacao.csv")
teste = pd.read_csv("dados_modelagem/teste.csv")

X_train = treino.drop(columns=["SK_ID_CURR", "TARGET"])
y_train = treino["TARGET"]
X_valid = validacao.drop(columns=["SK_ID_CURR", "TARGET"])
y_valid = validacao["TARGET"]
X_test = teste.drop(columns=["SK_ID_CURR"])
```

O notebook guiado anterior contém uma preparação alternativa a partir dos originais. Ao usar as matrizes prontas acima, siga diretamente para seus exercícios de treinamento, usando os nomes `X_train`, `X_valid`, `y_train`, `y_valid` e `X_test`. Não execute as duas preparações sobre a mesma matriz. Para gerar previsões finais após avaliar os modelos, ajuste novamente a preparação e os modelos em todo o treino rotulado.

As regras de limpeza são hipóteses documentadas no relatório, especialmente carro com 60 anos ou mais. Extremos de renda e crédito foram sinalizados e preservados. Os dados de teste não participaram do cálculo de medianas nem da definição das categorias. As associações exploratórias consultaram todo o treino rotulado; avaliações posteriores orientadas por essa EDA devem ser interpretadas como exploratórias.

## Reproduzir

Na pasta deste projeto, instale as dependências de `requirements_analise.txt` no seu ambiente e execute:

```powershell
python executar_notebook_base.py
```

Na máquina usada para a análise, o comando é:

```powershell
.\.venv_analise\Scripts\python.exe executar_notebook_base.py
```

Você também pode abrir o notebook e executar as células normalmente no Jupyter. `executar_notebook_base.py` executa as células em sequência com Python e incorpora as tabelas e figuras nas saídas, sem precisar iniciar um servidor Jupyter. A execução atualiza os CSVs tratados e preparados; os originais em `input/` são apenas lidos.

Para reproduzir o relatório detalhado anterior, use `executar_analise.py`, que lê `analise_dados.py` e grava `notebooks/referencia/analise_exploratoria_detalhada.ipynb`. Esse script não substitui o notebook base. As versões usadas na análise detalhada ficam em `analise/verificacoes.json` e as somas SHA-256 em `analise/manifesto_originais.csv`.

## Comparação de modelos em pares

O notebook base `notebooks/base/analise_exploratoria.ipynb` foi mantido intacto. As seis cópias acrescentam treinamento, avaliação e previsão após todas as células originais:

- `notebooks/ensembles/01_ensemble_adaboost_xgboost.ipynb`
- `notebooks/ensembles/02_ensemble_adaboost_catboost.ipynb`
- `notebooks/ensembles/03_ensemble_adaboost_lightgbm.ipynb`
- `notebooks/ensembles/04_ensemble_xgboost_catboost.ipynb`
- `notebooks/ensembles/05_ensemble_xgboost_lightgbm.ipynb`
- `notebooks/ensembles/06_ensemble_catboost_lightgbm.ipynb`

Todos usam a mesma divisão 80/20 e a média 50/50 das probabilidades. O resultado completo está em `resultados_ensembles/comparacao.md`, e o ranking numérico em `resultados_ensembles/ranking_combinacoes.csv`. A comparação inclui os quatro modelos individuais para conferir se a combinação melhora o melhor componente.

Cada algoritmo foi treinado uma vez, e seus modelos salvos foram reutilizados nos pares. Nos seis notebooks, `usar_modelos_salvos=True` reproduz a rodada usando esses arquivos; escolha `False` para treinar novamente. Ao mudar parâmetros, tratamento ou divisão, use `False`. A leitura de arquivos salvos só reproduz a configuração registrada. As células acrescentadas foram executadas; as células da base foram copiadas com suas saídas previamente executadas e conferidas.

O melhor par observado nesta rodada foi CatBoost + LightGBM, com ROC-AUC de validação 0,748595. Esse resultado é específico desta divisão e dos parâmetros usados. A comparação também seleciona o par pela mesma validação, portanto não é uma avaliação independente da seleção.

Após essa comparação, os dois modelos do melhor par foram ajustados em 100% do treino rotulado para gerar `resultados_ensembles/submission_melhor_par_treino_completo.csv`. Essa etapa aparece ao final do notebook do par vencedor. Os CSVs das pastas de cada par com sufixo `modelos_80` usam apenas o subtreino de 80%. O arquivo final não foi enviado ao Omnicampus.

Para refazer a rodada no ambiente local:

```powershell
.\.venv_analise\Scripts\python.exe -m pip install -r requirements_ensembles.txt
.\.venv_analise\Scripts\python.exe comparar_ensembles.py
.\.venv_analise\Scripts\python.exe criar_notebooks_ensembles.py
```

Esses dois scripts não executam nem modificam o notebook base. O primeiro usa as matrizes já preparadas, e o segundo copia a base e executa somente as células novas. A integridade da base é registrada por SHA-256 e a igualdade de todas as células copiadas em `resultados_ensembles/verificacao_notebooks.json`.
