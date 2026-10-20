"""Copia a base sem alterá-la e acrescenta células simples para cada par."""
from pathlib import Path
import ast
import base64
import contextlib
import copy
import hashlib
import io
import json
import traceback
import warnings
import nbformat
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados_ensembles"
arquivo_base = BASE / "notebooks" / "base" / "analise_exploratoria.ipynb"
hash_base = hashlib.sha256(arquivo_base.read_bytes()).hexdigest()
base_nb = nbformat.read(arquivo_base, as_version=4)
proveniencia = json.loads((SAIDA / "proveniencia.json").read_text(encoding="utf-8"))
assert hash_base == proveniencia["base_sha256"]
ranking = pd.read_csv(SAIDA / "ranking_combinacoes.csv")
melhor = json.loads((SAIDA / "melhor_par.json").read_text(encoding="utf-8"))

construtores = {
    "AdaBoost": "AdaBoostClassifier(\n    estimator=DecisionTreeClassifier(max_depth=2, random_state=42),\n    n_estimators=100, learning_rate=0.1, random_state=42\n)",
    "XGBoost": "XGBClassifier(\n    n_estimators=300, max_depth=3, learning_rate=0.05,\n    subsample=0.8, colsample_bytree=0.8, tree_method=\"hist\",\n    objective=\"binary:logistic\", eval_metric=\"auc\", random_state=42, n_jobs=2\n)",
    "CatBoost": "CatBoostClassifier(\n    iterations=300, depth=5, learning_rate=0.05,\n    loss_function=\"Logloss\", eval_metric=\"AUC\", random_seed=42,\n    thread_count=2, verbose=False, allow_writing_files=False\n)",
    "LightGBM": "LGBMClassifier(\n    n_estimators=300, num_leaves=15, learning_rate=0.05,\n    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,\n    random_state=42, n_jobs=2, verbosity=-1\n)"
}

# Reutilizar as matrizes já executadas na base para conferir somente o acréscimo.
treino_limpo = pd.read_csv(BASE / "dados_tratados" / "train.csv")
teste_limpo = pd.read_csv(BASE / "dados_tratados" / "test.csv")
dados_treino = pd.read_csv(BASE / "dados_modelagem" / "treino.csv")
dados_validacao = pd.read_csv(BASE / "dados_modelagem" / "validacao.csv")
dados_teste = pd.read_csv(BASE / "dados_modelagem" / "teste.csv")
indice_por_id = pd.Series(treino_limpo.index, index=treino_limpo.SK_ID_CURR)
dados_treino.index = indice_por_id.loc[dados_treino.SK_ID_CURR].to_numpy()
dados_validacao.index = indice_por_id.loc[dados_validacao.SK_ID_CURR].to_numpy()
ambiente_base = {
    "__name__": "__main__", "pd": pd, "np": np, "plt": plt, "Path": Path,
    "pasta": BASE, "treino_limpo": treino_limpo, "teste_limpo": teste_limpo,
    "X_treino": dados_treino.drop(columns=["SK_ID_CURR", "TARGET"]),
    "X_validacao": dados_validacao.drop(columns=["SK_ID_CURR", "TARGET"]),
    "X_teste": dados_teste.drop(columns="SK_ID_CURR"),
    "y_treino": dados_treino.TARGET, "y_validacao": dados_validacao.TARGET,
    "numericas": treino_limpo.drop(columns=["SK_ID_CURR", "TARGET"]).select_dtypes(include="number").columns.tolist(),
    "categoricas": treino_limpo.select_dtypes(exclude="number").columns.tolist()
}
plt.show = lambda: None
verificacoes = []

for _, linha in ranking.sort_values("notebook").iterrows():
    a, b, slug = linha.modelo_1, linha.modelo_2, linha.slug
    nb = copy.deepcopy(base_nb)
    adicionais = []
    adicionais.append(nbformat.v4.new_markdown_cell(
        f"# Parte 4 — Ensemble {a} + {b}\n\n"
        "Todas as células anteriores são uma cópia integral do notebook base, com sua divisão 80/20 e o mesmo tratamento. "
        "A base original não foi editada. Agora começa a previsão desta combinação.\n\n"
        "Vamos avaliar cada modelo separadamente e depois a média 50/50 das probabilidades. "
        "A métrica principal é ROC-AUC na validação. A média pode melhorar ou piorar o melhor modelo individual."))
    adicionais.append(nbformat.v4.new_markdown_cell(
        "## 29. Bibliotecas dos modelos\n\n"
        "Neste ambiente elas já estão instaladas. Em outro ambiente, instale as bibliotecas antes de executar o notebook, por exemplo:\n\n"
        "```python\n%pip install scikit-learn xgboost catboost lightgbm joblib\n```\n\n"
        "Os próximos blocos usam classes prontas das bibliotecas, sem funções próprias."))
    adicionais.append(nbformat.v4.new_code_cell(
        "import joblib\nfrom sklearn.ensemble import AdaBoostClassifier\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "from sklearn.metrics import roc_auc_score, roc_curve\n"
        "from xgboost import XGBClassifier\nfrom catboost import CatBoostClassifier\n"
        "from lightgbm import LGBMClassifier"))
    adicionais.append(nbformat.v4.new_markdown_cell(
        "## 30. Usar as mesmas entradas\n\n"
        "Os modelos receberão as mesmas colunas e linhas preparadas pela base. "
        "to_numpy troca apenas o formato da tabela e preserva a ordem das colunas. "
        "Usamos float32 para padronizar a entrada e evitar restrições de caracteres nos nomes das colunas no LightGBM."))
    adicionais.append(nbformat.v4.new_code_cell(
        "treino_modelo = X_treino.to_numpy(dtype=\"float32\")\n"
        "validacao_modelo = X_validacao.to_numpy(dtype=\"float32\")\n"
        "teste_modelo = X_teste.to_numpy(dtype=\"float32\")\n\n"
        "print(treino_modelo.shape, validacao_modelo.shape, teste_modelo.shape)"))
    adicionais.append(nbformat.v4.new_markdown_cell(
        "## 31. Treinar ou usar os modelos desta rodada\n\n"
        "Para comparar os seis pares, cada algoritmo foi treinado uma vez na mesma base. "
        "Os modelos locais salvos são reutilizados nos pares para evitar treinar o mesmo algoritmo três vezes.\n\n"
        "Com usar_modelos_salvos=True, vamos carregar esses modelos. "
        "Para acompanhar o treinamento do zero, escolha False. **Se alterar parâmetros, tratamento ou divisão, escolha False**. "
        "Os arquivos salvos só correspondem à configuração registrada desta rodada. "
        "Os modelos de validação usam 80% dos dados rotulados; o teste sem rótulos não participa do ajuste."))
    adicionais.append(nbformat.v4.new_code_cell(
        f"usar_modelos_salvos = True  # False para treinar novamente.\n"
        f"pasta_modelos = pasta / \"resultados_ensembles\" / \"modelos\"\n"
        f"pasta_resultado = pasta / \"resultados_ensembles\" / \"{slug}\"\n"
        "pasta_resultado.mkdir(parents=True, exist_ok=True)"))
    adicionais.append(nbformat.v4.new_markdown_cell(
        f"### Primeiro modelo: {a}\n\n"
        "A quantidade de árvores e a profundidade controlam a complexidade. "
        "learning_rate controla a contribuição das etapas do boosting. Os valores abaixo são um ponto de partida fixo; "
        "não fizemos busca de hiperparâmetros nesta comparação."))
    adicionais.append(nbformat.v4.new_code_cell(
        f"modelo_1 = {construtores[a]}\n\n"
        f"if usar_modelos_salvos:\n    modelo_1 = joblib.load(pasta_modelos / \"{a.lower()}.joblib\")\n"
        "else:\n    modelo_1.fit(treino_modelo, y_treino)\n\n"
        f"print(\"{a} pronto.\")"))
    adicionais.append(nbformat.v4.new_markdown_cell(f"### Segundo modelo: {b}\n\nAgora repetimos o processo para o segundo algoritmo usando a mesma parte de treino."))
    adicionais.append(nbformat.v4.new_code_cell(
        f"modelo_2 = {construtores[b]}\n\n"
        f"if usar_modelos_salvos:\n    modelo_2 = joblib.load(pasta_modelos / \"{b.lower()}.joblib\")\n"
        "else:\n    modelo_2.fit(treino_modelo, y_treino)\n\n"
        f"print(\"{b} pronto.\")"))
    adicionais.append(nbformat.v4.new_markdown_cell(
        "## 32. Fazer previsões de probabilidade\n\n"
        "predict_proba retorna probabilidades das classes 0 e 1. Selecionamos [:, 1] para dificuldades de pagamento. "
        "Vamos medir também o treino, para observar uma possível diferença para a validação."))
    adicionais.append(nbformat.v4.new_code_cell(
        "prob_1_treino = modelo_1.predict_proba(treino_modelo)[:, 1]\n"
        "prob_2_treino = modelo_2.predict_proba(treino_modelo)[:, 1]\n"
        "prob_1_validacao = modelo_1.predict_proba(validacao_modelo)[:, 1]\n"
        "prob_2_validacao = modelo_2.predict_proba(validacao_modelo)[:, 1]"))
    adicionais.append(nbformat.v4.new_markdown_cell(
        "## 33. Combinar as probabilidades\n\n"
        "A regra é uma média simples. Cada modelo tem coeficiente 0,5. "
        "Não ajustamos os pesos para tentar aumentar a nota nesta validação. "
        "Pesos iguais não garantem influência igual na ordenação, porque as probabilidades dos modelos podem ter escalas diferentes."))
    adicionais.append(nbformat.v4.new_code_cell(
        "prob_ensemble_treino = (prob_1_treino + prob_2_treino) / 2\n"
        "prob_ensemble_validacao = (prob_1_validacao + prob_2_validacao) / 2"))
    adicionais.append(nbformat.v4.new_markdown_cell("## 34. Comparar os três resultados\n\nO ensemble deve ser comparado com os modelos individuais. Uma combinação com nota maior que outra combinação ainda pode perder para um modelo sozinho."))
    adicionais.append(nbformat.v4.new_code_cell(
        f"resultados = pd.DataFrame({{\n    \"Modelo\": [\"{a}\", \"{b}\", \"{a} + {b} (50/50)\"],\n"
        "    \"ROC-AUC treino\": [\n        roc_auc_score(y_treino, prob_1_treino),\n"
        "        roc_auc_score(y_treino, prob_2_treino),\n        roc_auc_score(y_treino, prob_ensemble_treino)\n    ],\n"
        "    \"ROC-AUC validação\": [\n        roc_auc_score(y_validacao, prob_1_validacao),\n"
        "        roc_auc_score(y_validacao, prob_2_validacao),\n        roc_auc_score(y_validacao, prob_ensemble_validacao)\n    ]\n})\n"
        "resultados.sort_values(\"ROC-AUC validação\", ascending=False)"))
    adicionais.append(nbformat.v4.new_code_cell(
        "melhor_individual = max(roc_auc_score(y_validacao, prob_1_validacao),\n"
        "                        roc_auc_score(y_validacao, prob_2_validacao))\n"
        "ganho = roc_auc_score(y_validacao, prob_ensemble_validacao) - melhor_individual\n"
        "print(\"Ganho do ensemble sobre o melhor individual:\", round(ganho, 6))"))
    adicionais.append(nbformat.v4.new_markdown_cell("## 35. Ver as curvas ROC\n\nAs três curvas usam exatamente as mesmas linhas de validação."))
    adicionais.append(nbformat.v4.new_code_cell(
        "plt.figure(figsize=(7, 5))\n"
        f"for nome, prob in [(\"{a}\", prob_1_validacao), (\"{b}\", prob_2_validacao),\n"
        "                   (\"Ensemble 50/50\", prob_ensemble_validacao)]:\n"
        "    falsos_positivos, verdadeiros_positivos, limites = roc_curve(y_validacao, prob)\n"
        "    plt.plot(falsos_positivos, verdadeiros_positivos, label=nome)\n"
        "plt.plot([0, 1], [0, 1], linestyle=\"--\", color=\"gray\")\n"
        "plt.xlabel(\"Taxa de falsos positivos\")\nplt.ylabel(\"Taxa de verdadeiros positivos\")\n"
        "plt.title(\"Curvas ROC na validação\")\nplt.legend()\nplt.show()"))
    adicionais.append(nbformat.v4.new_markdown_cell(
        "## 36. Salvar os resultados desta combinação\n\n"
        "Guardamos os IDs e o alvo junto às probabilidades para permitir conferir a avaliação. "
        "As previsões do teste abaixo usam os modelos de 80%; elas não têm AUC medido, porque os rótulos não estão disponíveis."))
    adicionais.append(nbformat.v4.new_code_cell(
        "previsoes_validacao = pd.DataFrame({\n"
        "    \"SK_ID_CURR\": treino_limpo.loc[X_validacao.index, \"SK_ID_CURR\"],\n"
        "    \"TARGET\": y_validacao,\n    \"prob_modelo_1\": prob_1_validacao,\n"
        "    \"prob_modelo_2\": prob_2_validacao,\n    \"prob_ensemble\": prob_ensemble_validacao\n})\n"
        "previsoes_validacao.to_csv(pasta_resultado / \"validacao.csv\", index=False)\n"
        "resultados.to_csv(pasta_resultado / \"metricas.csv\", index=False)"))
    adicionais.append(nbformat.v4.new_code_cell(
        "prob_1_teste = modelo_1.predict_proba(teste_modelo)[:, 1]\n"
        "prob_2_teste = modelo_2.predict_proba(teste_modelo)[:, 1]\n"
        "previsoes_teste = pd.DataFrame({\n    \"SK_ID_CURR\": teste_limpo[\"SK_ID_CURR\"],\n"
        "    \"TARGET\": (prob_1_teste + prob_2_teste) / 2\n})\n"
        "previsoes_teste.to_csv(pasta_resultado / \"previsoes_teste_modelos_80.csv\", index=False)\n"
        "previsoes_teste.head()"))
    adicionais.append(nbformat.v4.new_markdown_cell(
        "## 37. Ver o ranking da rodada\n\n"
        "O ranking registrado reúne seis pares com os mesmos pesos, divisão e parâmetros. "
        "Se você alterar parâmetros ou pesos para estudar, registre um novo experimento e refaça todos os pares antes de atualizar a comparação. "
        "Escolher o maior AUC nesta validação não garante melhor resultado em outra amostra."))
    adicionais.append(nbformat.v4.new_code_cell(
        "ranking_rodada = pd.read_csv(pasta / \"resultados_ensembles\" / \"ranking_combinacoes.csv\")\n"
        "ranking_rodada[[\"posicao\", \"par\", \"roc_auc_validacao\", \"ganho_sobre_melhor_individual\"]]"))

    if slug == melhor["slug"]:
        adicionais.append(nbformat.v4.new_markdown_cell(
            "# Parte 5 — Ajuste final do melhor par em todo o treino\n\n"
            "Este foi o melhor par observado na validação desta rodada. Agora refazemos a preparação com todas as linhas rotuladas, "
            "preservando a mesma regra de tratamento. Depois ajustamos os dois modelos novamente e prevemos o teste. "
            "O AUC da Parte 4 corresponde aos modelos anteriores de 80%; não é uma nota deste ajuste final."))
        adicionais.append(nbformat.v4.new_code_cell(
            "X_completo = treino_limpo.drop(columns=[\"SK_ID_CURR\", \"TARGET\"]).copy()\n"
            "X_envio = teste_limpo.drop(columns=\"SK_ID_CURR\").copy()\n\n"
            "for coluna in numericas:\n    X_completo[coluna + \"_AUSENTE\"] = X_completo[coluna].isna().astype(int)\n"
            "    X_envio[coluna + \"_AUSENTE\"] = X_envio[coluna].isna().astype(int)\n\n"
            "medianas_finais = X_completo[numericas].median()\n"
            "X_completo[numericas] = X_completo[numericas].fillna(medianas_finais)\n"
            "X_envio[numericas] = X_envio[numericas].fillna(medianas_finais)"))
        adicionais.append(nbformat.v4.new_code_cell(
            "X_completo[categoricas] = X_completo[categoricas].fillna(\"Ausente\")\n"
            "X_envio[categoricas] = X_envio[categoricas].fillna(\"Ausente\")\n"
            "X_completo = pd.get_dummies(X_completo, columns=categoricas, dtype=int)\n"
            "X_envio = pd.get_dummies(X_envio, columns=categoricas, dtype=int)\n"
            "X_envio = X_envio.reindex(columns=X_completo.columns, fill_value=0)\n"
            "constantes_finais = X_completo.columns[X_completo.nunique() <= 1].tolist()\n"
            "X_completo = X_completo.drop(columns=constantes_finais)\n"
            "X_envio = X_envio.drop(columns=constantes_finais)"))
        adicionais.append(nbformat.v4.new_code_cell(
            f"modelo_1_final = {construtores[a]}\n"
            f"modelo_2_final = {construtores[b]}\n\n"
            "if usar_modelos_salvos:\n"
            f"    modelo_1_final = joblib.load(pasta / \"resultados_ensembles\" / \"modelos_finais\" / \"{a.lower()}.joblib\")\n"
            f"    modelo_2_final = joblib.load(pasta / \"resultados_ensembles\" / \"modelos_finais\" / \"{b.lower()}.joblib\")\n"
            "else:\n    modelo_1_final.fit(X_completo.to_numpy(dtype=\"float32\"), treino_limpo[\"TARGET\"])\n"
            "    modelo_2_final.fit(X_completo.to_numpy(dtype=\"float32\"), treino_limpo[\"TARGET\"])"))
        adicionais.append(nbformat.v4.new_code_cell(
            "prob_1_final = modelo_1_final.predict_proba(X_envio.to_numpy(dtype=\"float32\"))[:, 1]\n"
            "prob_2_final = modelo_2_final.predict_proba(X_envio.to_numpy(dtype=\"float32\"))[:, 1]\n"
            "submission_final = pd.read_csv(pasta / \"input\" / \"sample_submission.csv\")\n"
            "submission_final[\"TARGET\"] = (prob_1_final + prob_2_final) / 2\n"
            "submission_final.to_csv(pasta / \"resultados_ensembles\" / \"submission_melhor_par_treino_completo.csv\", index=False)\n"
            "submission_final.head()"))
    adicionais.append(nbformat.v4.new_markdown_cell(
        "## Observações e referências\n\n"
        "Os resultados dependem desta divisão e destes parâmetros. O ranking é exploratório, pois a validação também selecionou o par. "
        "Modelos individuais podem superar a média. Em especial, AdaBoost SAMME pode produzir probabilidades com distribuição diferente dos demais, "
        "então 50/50 não implica contribuições iguais no ranking das previsões. "
        "Não usamos CatBoost com suas categorias nativas nesta rodada: ele recebe a mesma codificação dos demais modelos para preservar a comparação da base.\n\n"
        "Fontes: [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), "
        "[XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html), "
        "[CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier), "
        "[LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html). "
        "Dados e regras: materiais do curso registrados na base. Assistência: ChatGPT/Codex."))

    for i, celula in enumerate(adicionais, 1):
        celula.id = f"{slug}-{i:03d}"
    nb.cells.extend(adicionais)
    caminho = BASE / "notebooks" / "ensembles" / linha.notebook
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ambiente = dict(ambiente_base)
    numero_execucao = max(c.get("execution_count") or 0 for c in base_nb.cells)
    print("Conferindo", linha.notebook, flush=True)
    for celula in nb.cells[len(base_nb.cells):]:
        if celula.cell_type != "code":
            continue
        numero_execucao += 1
        captura = io.StringIO()
        outputs = []
        resultado = None
        erro = None
        arvore = ast.parse(celula.source)
        ultima = ast.Expression(arvore.body.pop().value) if arvore.body and isinstance(arvore.body[-1], ast.Expr) else None
        try:
            with contextlib.redirect_stdout(captura):
                exec(compile(arvore, str(caminho), "exec"), ambiente)
                if ultima is not None:
                    resultado = eval(compile(ultima, str(caminho), "eval"), ambiente)
        except Exception:
            erro = traceback.format_exc()
        if captura.getvalue():
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captura.getvalue()))
        if resultado is not None:
            dados = {"text/plain": repr(resultado)}
            if hasattr(resultado, "_repr_html_") and resultado._repr_html_():
                dados["text/html"] = resultado._repr_html_()
            outputs.append(nbformat.v4.new_output("execute_result", execution_count=numero_execucao, data=dados))
        for figura in plt.get_fignums():
            imagem = io.BytesIO()
            plt.figure(figura).savefig(imagem, format="png", bbox_inches="tight", dpi=110)
            outputs.append(nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(imagem.getvalue()).decode("ascii")}))
        plt.close("all")
        if erro:
            outputs.append(nbformat.v4.new_output("error", ename="ExecutionError", evalue=erro, traceback=erro.splitlines()))
        celula.outputs = outputs
        celula.execution_count = numero_execucao
        if erro:
            nbformat.write(nb, caminho)
            raise RuntimeError(erro)
    assert nb.cells[:len(base_nb.cells)] == base_nb.cells
    nbformat.validate(nb)
    nbformat.write(nb, caminho)
    assert nbformat.read(caminho, as_version=4).cells[:len(base_nb.cells)] == base_nb.cells
    medido = pd.read_csv(SAIDA / slug / "metricas.csv")
    referencia = float(linha.roc_auc_validacao)
    medido_par = medido.loc[medido.Modelo.eq(f"{a} + {b} (50/50)"), "ROC-AUC validação"].iloc[0]
    assert np.isclose(medido_par, referencia, rtol=0, atol=1e-12)
    verificacoes.append({"notebook": caminho.name, "base_copiada_integralmente": True,
        "acrescimo_executado": True, "auc_conferido": True})
assert hashlib.sha256(arquivo_base.read_bytes()).hexdigest() == hash_base
(SAIDA / "verificacao_notebooks.json").write_text(json.dumps({
    "base_sha256_antes": hash_base, "base_sha256_depois": hashlib.sha256(arquivo_base.read_bytes()).hexdigest(),
    "base_intacta": True, "celulas_base_por_copia": len(base_nb.cells), "notebooks": verificacoes
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Seis notebooks concluídos e conferidos. Base intacta.", flush=True)
