"""Treina quatro modelos na mesma base e avalia seis médias 50/50."""
from pathlib import Path
import hashlib
import itertools
import json
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
import xgboost
import catboost
import lightgbm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados_ensembles"
MODELOS = SAIDA / "modelos"
FINAIS = SAIDA / "modelos_finais"
for pasta in [SAIDA, MODELOS, FINAIS]:
    pasta.mkdir(exist_ok=True)
arquivo_base = BASE / "notebooks" / "base" / "analise_exploratoria.ipynb"
hash_base = hashlib.sha256(arquivo_base.read_bytes()).hexdigest()
treino = pd.read_csv(BASE / "dados_modelagem" / "treino.csv")
validacao = pd.read_csv(BASE / "dados_modelagem" / "validacao.csv")
teste = pd.read_csv(BASE / "dados_modelagem" / "teste.csv")
sample = pd.read_csv(BASE / "input" / "sample_submission.csv")
colunas = treino.columns.drop(["SK_ID_CURR", "TARGET"]).tolist()
assert colunas == validacao.columns.drop(["SK_ID_CURR", "TARGET"]).tolist()
assert colunas == teste.columns.drop("SK_ID_CURR").tolist()
assert len(treino) == 136961 and len(validacao) == 34241 and len(teste) == 61500
assert set(treino.SK_ID_CURR).isdisjoint(validacao.SK_ID_CURR)
assert sample.SK_ID_CURR.tolist() == teste.SK_ID_CURR.tolist()
X_treino = treino[colunas].to_numpy(dtype="float32")
X_validacao = validacao[colunas].to_numpy(dtype="float32")
X_teste = teste[colunas].to_numpy(dtype="float32")
y_treino = treino.TARGET.to_numpy()
y_validacao = validacao.TARGET.to_numpy()
for matriz in [X_treino, X_validacao, X_teste]:
    assert np.isfinite(matriz).all()

# As mesmas configurações são usadas em todos os pares.
modelos = {
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
        n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        objective="binary:logistic", eval_metric="auc", random_state=42, n_jobs=2),
    "CatBoost": CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05, loss_function="Logloss",
        eval_metric="AUC", random_seed=42, thread_count=2,
        verbose=False, allow_writing_files=False),
    "LightGBM": LGBMClassifier(
        n_estimators=300, num_leaves=15, learning_rate=0.05,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, n_jobs=2, verbosity=-1)
}
versoes = {"python": platform.python_version(), "numpy": np.__version__,
    "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    "xgboost": xgboost.__version__, "catboost": catboost.__version__,
    "lightgbm": lightgbm.__version__}
proveniencia = {"base": arquivo_base.name, "base_sha256": hash_base,
    "divisao": "80/20 estratificada; random_state=42", "treino": len(treino),
    "validacao": len(validacao), "teste": len(teste), "preditores": len(colunas),
    "versoes": versoes, "pesos": [0.5, 0.5],
    "parametros": {nome: json.loads(json.dumps(modelo.get_params(), default=str))
                   for nome, modelo in modelos.items()}, "dados_sha256": {}}
for nome in ["treino.csv", "validacao.csv", "teste.csv"]:
    proveniencia["dados_sha256"][nome] = hashlib.sha256((BASE / "dados_modelagem" / nome).read_bytes()).hexdigest()
anterior = json.loads((SAIDA / "proveniencia.json").read_text(encoding="utf-8")) if (SAIDA / "proveniencia.json").exists() else None
reutilizar = anterior == proveniencia
(SAIDA / "proveniencia.json").write_text(json.dumps(proveniencia, ensure_ascii=False, indent=2), encoding="utf-8")
print("Base preservada:", hash_base, flush=True)
print("Dimensões:", X_treino.shape, X_validacao.shape, X_teste.shape, flush=True)

pred_treino = {}
pred_validacao = {}
pred_teste = {}
individuais = []
for nome, modelo in modelos.items():
    inicio = time.perf_counter()
    arquivo = MODELOS / f"{nome.lower()}.joblib"
    print("Treinando", nome, flush=True)
    if reutilizar and arquivo.exists():
        modelo = joblib.load(arquivo)
        modelos[nome] = modelo
    else:
        modelo.fit(X_treino, y_treino)
        joblib.dump(modelo, arquivo)
    assert modelo.classes_.tolist() == [0, 1]
    pred_treino[nome] = modelo.predict_proba(X_treino)[:, 1]
    pred_validacao[nome] = modelo.predict_proba(X_validacao)[:, 1]
    pred_teste[nome] = modelo.predict_proba(X_teste)[:, 1]
    for prob in [pred_treino[nome], pred_validacao[nome], pred_teste[nome]]:
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    linha = {"modelo": nome, "roc_auc_treino": roc_auc_score(y_treino, pred_treino[nome]),
        "roc_auc_validacao": roc_auc_score(y_validacao, pred_validacao[nome]),
        "segundos_ajuste_e_previsoes": time.perf_counter() - inicio}
    individuais.append(linha)
    pd.DataFrame(individuais).to_csv(SAIDA / "modelos_individuais.csv", index=False)
    print(json.dumps(linha), flush=True)
individuais = pd.DataFrame(individuais).sort_values("roc_auc_validacao", ascending=False)
pd.DataFrame({"SK_ID_CURR": validacao.SK_ID_CURR, "TARGET": y_validacao,
              **pred_validacao}).to_csv(SAIDA / "previsoes_individuais_validacao.csv", index=False)

pares = []
for numero, (a, b) in enumerate(itertools.combinations(modelos, 2), 1):
    slug = f"{a.lower()}_{b.lower()}"
    pasta = SAIDA / slug
    pasta.mkdir(exist_ok=True)
    prob_val = (pred_validacao[a] + pred_validacao[b]) / 2
    prob_train = (pred_treino[a] + pred_treino[b]) / 2
    prob_test = (pred_teste[a] + pred_teste[b]) / 2
    individuais_par = individuais.set_index("modelo").loc[[a, b]]
    melhor_individual = individuais_par.roc_auc_validacao.max()
    auc_val = roc_auc_score(y_validacao, prob_val)
    linha = {"par": f"{a} + {b}", "modelo_1": a, "modelo_2": b, "slug": slug,
        "notebook": f"{numero:02d}_ensemble_{slug}.ipynb", "peso_1": 0.5, "peso_2": 0.5,
        "roc_auc_treino": roc_auc_score(y_treino, prob_train), "roc_auc_validacao": auc_val,
        "melhor_individual_no_par": melhor_individual,
        "ganho_sobre_melhor_individual": auc_val - melhor_individual}
    pares.append(linha)
    pd.DataFrame({"SK_ID_CURR": validacao.SK_ID_CURR, "TARGET": y_validacao,
        "prob_modelo_1": pred_validacao[a], "prob_modelo_2": pred_validacao[b],
        "prob_ensemble": prob_val}).to_csv(pasta / "validacao.csv", index=False)
    envio = sample.copy()
    envio["TARGET"] = prob_test
    envio.to_csv(pasta / "previsoes_teste_modelos_80.csv", index=False)
    metricas = pd.DataFrame({"Modelo": [a, b, f"{a} + {b} (50/50)"],
        "ROC-AUC treino": [roc_auc_score(y_treino, pred_treino[a]), roc_auc_score(y_treino, pred_treino[b]), linha["roc_auc_treino"]],
        "ROC-AUC validação": [roc_auc_score(y_validacao, pred_validacao[a]), roc_auc_score(y_validacao, pred_validacao[b]), auc_val]})
    metricas.to_csv(pasta / "metricas.csv", index=False)
    print("Par", linha["par"], "AUC", auc_val, flush=True)
ranking = pd.DataFrame(pares).sort_values("roc_auc_validacao", ascending=False).reset_index(drop=True)
ranking.insert(0, "posicao", np.arange(1, len(ranking) + 1))
ranking.to_csv(SAIDA / "ranking_combinacoes.csv", index=False)
melhor = ranking.iloc[0].to_dict()
(SAIDA / "melhor_par.json").write_text(json.dumps(melhor, ensure_ascii=False, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(10, 5))
ranking.iloc[::-1].plot.barh(x="par", y="roc_auc_validacao", ax=ax, legend=False, color="#246c96")
ax.set(xlabel="ROC-AUC de validação", ylabel="", title="Seis pares — mesma validação 20%, média 50/50")
ax.set_xlim(max(0.5, ranking.roc_auc_validacao.min() - 0.015), ranking.roc_auc_validacao.max() + 0.01)
for i, (_, linha) in enumerate(ranking.iloc[::-1].iterrows()):
    ax.text(linha.roc_auc_validacao + 0.0004, i, f"{linha.roc_auc_validacao:.5f}", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(SAIDA / "ranking.png", dpi=140)
plt.close(fig)

# O teste sem rótulos não participa da escolha. Só agora ajustamos o melhor par em 100%.
print("Melhor par observado:", melhor["par"], "— preparando ajuste final", flush=True)
treino_limpo = pd.read_csv(BASE / "dados_tratados" / "train.csv")
teste_limpo = pd.read_csv(BASE / "dados_tratados" / "test.csv")
X_completo = treino_limpo.drop(columns=["SK_ID_CURR", "TARGET"]).copy()
X_envio = teste_limpo.drop(columns="SK_ID_CURR").copy()
numericas = X_completo.select_dtypes(include="number").columns.tolist()
categoricas = X_completo.select_dtypes(exclude="number").columns.tolist()
for dados in [X_completo, X_envio]:
    for coluna in numericas:
        dados[coluna + "_AUSENTE"] = dados[coluna].isna().astype(int)
    dados[categoricas] = dados[categoricas].fillna("Ausente")
medianas_finais = X_completo[numericas].median()
assert medianas_finais.notna().all()
X_completo[numericas] = X_completo[numericas].fillna(medianas_finais)
X_envio[numericas] = X_envio[numericas].fillna(medianas_finais)
X_completo = pd.get_dummies(X_completo, columns=categoricas, dtype=int)
X_envio = pd.get_dummies(X_envio, columns=categoricas, dtype=int).reindex(columns=X_completo.columns, fill_value=0)
constantes_finais = X_completo.columns[X_completo.nunique().le(1)].tolist()
X_completo = X_completo.drop(columns=constantes_finais)
X_envio = X_envio.drop(columns=constantes_finais)
(FINAIS / "colunas.json").write_text(json.dumps(X_completo.columns.tolist(), indent=2), encoding="utf-8")
medianas_finais.to_csv(FINAIS / "medianas.csv", header=["mediana"])
probs_finais = []
for nome in [melhor["modelo_1"], melhor["modelo_2"]]:
    print("Ajuste final 100%:", nome, flush=True)
    modelo_final = clone(modelos[nome])
    modelo_final.fit(X_completo.to_numpy(dtype="float32"), treino_limpo.TARGET)
    joblib.dump(modelo_final, FINAIS / f"{nome.lower()}.joblib")
    probs_finais.append(modelo_final.predict_proba(X_envio.to_numpy(dtype="float32"))[:, 1])
final = sample.copy()
assert final.SK_ID_CURR.tolist() == teste_limpo.SK_ID_CURR.tolist()
final["TARGET"] = (probs_finais[0] + probs_finais[1]) / 2
assert final.TARGET.notna().all() and final.TARGET.between(0, 1).all()
assert len(final) == 61500
final.to_csv(SAIDA / "submission_melhor_par_treino_completo.csv", index=False)
assert hashlib.sha256(arquivo_base.read_bytes()).hexdigest() == hash_base

relatorio = ["# Comparação dos seis ensembles", "",
    f"Melhor par observado: **{melhor['par']}**, ROC-AUC de validação **{melhor['roc_auc_validacao']:.6f}**.", "",
    "## Protocolo", "",
    "Notebook base preservado integralmente, incluindo células, código e saídas. Cada notebook de combinação começa com uma cópia dessas células e acrescenta a previsão. "
    "Todos os pares usam as mesmas 136.961 linhas de treino, 34.241 linhas de validação e 141 preditores. Divisão estratificada 80/20, semente 42. "
    "Nenhum dado externo foi usado. Medianas e one-hot são os mesmos da base. As matrizes viram arrays float32 na mesma ordem para evitar restrições de caracteres dos nomes no LightGBM.", "",
    "Cada algoritmo foi treinado uma vez e reutilizado nos três pares em que participa. As configurações são fixas por algoritmo; "
    "não houve busca de parâmetros, otimização de pesos, calibração ou early stopping na validação. "
    "A média é sempre (probabilidade 1 + probabilidade 2) / 2. Os quatro modelos individuais também são comparados.", "",
    "## Ranking dos pares", "",
    ranking[["posicao", "par", "roc_auc_treino", "roc_auc_validacao", "melhor_individual_no_par", "ganho_sobre_melhor_individual"]].round(6).to_markdown(index=False), "",
    "![Ranking](ranking.png)", "", "## Modelos individuais", "", individuais.round(6).to_markdown(index=False), "",
    f"Melhor modelo individual: **{individuais.iloc[0]['modelo']}**, AUC **{individuais.iloc[0]['roc_auc_validacao']:.6f}**. "
    f"Diferença do melhor par para esse modelo: **{melhor['roc_auc_validacao'] - individuais.iloc[0]['roc_auc_validacao']:+.6f}**.", "",
    "Uma média pode piorar o melhor modelo individual. Pesos iguais não garantem contribuições iguais na ordenação, "
    "pois as distribuições das probabilidades podem diferir, especialmente com AdaBoost SAMME. "
    "Este é o melhor par apenas entre as seis médias 50/50 com os parâmetros e a divisão usados, não uma prova de ótimo global.", "",
    "A mesma validação foi usada para selecionar o par. O ranking é exploratório e pode variar em outra divisão. "
    "Os rótulos do teste não estão disponíveis, portanto nenhum score de teste foi medido nem se garante posição no leaderboard.", "",
    "## Previsões", "",
    "Cada pasta de par contém metricas.csv, validacao.csv com IDs/rótulos/probabilidades e previsoes_teste_modelos_80.csv. "
    "Essas previsões de teste usam modelos treinados em 80% dos dados rotulados. "
    "Após escolher o melhor par pela validação, refizemos a imputação, a codificação e o treinamento desse par com 100% do treino. "
    "O resultado é submission_melhor_par_treino_completo.csv, com 61.500 IDs e probabilidades. O AUC de validação do ranking se refere aos modelos de 80%; "
    "não foi atribuído ao ajuste de 100%. Nenhum arquivo foi enviado ao Omnicampus.", "",
    "## Reprodução", "",
    "Abra qualquer um dos seis notebooks na pasta GCI. A seção inicial é a cópia integral da base. "
    "Nas células acrescentadas, usar_modelos_salvos=True reproduz a rodada com os modelos já ajustados; "
    "defina False para treinar os dois modelos novamente. Se alterar tratamento, divisão ou parâmetros, use False. "
    "Para refazer toda a comparação com a configuração registrada, execute comparar_ensembles.py e depois criar_notebooks_ensembles.py. "
    "Os arquivos .joblib são artefatos locais produzidos nesta execução e dependem das versões registradas em proveniencia.json.", "",
    "## Referências", "",
    "- [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)\n"
    "- [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html)\n"
    "- [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)\n"
    "- [LGBMClassifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)", "",
    "Elaborado com assistência de ChatGPT/Codex. Fontes dos dados e regras: materiais da competição já registrados no notebook base."]
(SAIDA / "comparacao.md").write_text("\n".join(relatorio), encoding="utf-8")
print(ranking[["par", "roc_auc_validacao", "ganho_sobre_melhor_individual"]].to_string(index=False), flush=True)
print("Comparação e previsão final concluídas; SHA-256 da base conferido.", flush=True)
