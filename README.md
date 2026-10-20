# GCI — Home Credit Default Risk

Projeto de análise e modelagem para a competição do GCI World 2026 September.

## Guia do curso

O [GCI World 2026 September — Student Guide](https://curved-rambutan-a71.notion.site/GCI-World-2026-September-Student-Guide-971cfa7cece78245925781c15c64559b), no Notion, é o guia de referência do curso relacionado a este projeto.

## Notebooks

- [Base — análise exploratória e preparação](notebooks/base/analise_exploratoria.ipynb): notebook independente, sem treinamento de ensembles.
- [Ensembles](notebooks/ensembles/): seis notebooks, um por combinação de AdaBoost, XGBoost, CatBoost e LightGBM.
- [Referências](notebooks/referencia/): análise detalhada anterior e roteiro de estudo AdaBoost + XGBoost.

Os notebooks encontram a raiz do projeto a partir da pasta atual e suas pastas superiores. Para Colab, copie os dados oficiais para seu Drive e ajuste o caminho indicado na célula inicial.

## Dados e execução

Baixe os dados dos materiais oficiais do curso e coloque `train.csv`, `test.csv` e `sample_submission.csv` em `input/`. Dados, modelos treinados, previsões e anotações pessoais ficam locais e não são versionados.

Instale `requirements_analise.txt` e, para os modelos, `requirements_ensembles.txt`. Execute os scripts a partir da raiz:

```powershell
python executar_notebook_base.py
python comparar_ensembles.py
python criar_notebooks_ensembles.py
```

Para treinar os ensembles sem modelos locais salvos, ajuste `usar_modelos_salvos=False` na célula correspondente do notebook. Os resultados de execuções anteriores são ilustrativos e não comprovam colocação na competição.

Consulte [LEIA_ME.md](LEIA_ME.md) para detalhes da preparação, validação e reprodução.
