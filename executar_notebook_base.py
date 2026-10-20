"""Executa as células do notebook base e grava suas saídas, sem treinar modelos."""
from pathlib import Path
import ast
import base64
import contextlib
import io
import json
import os
import sys
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

base = Path(__file__).resolve().parent
os.chdir(base)
caminho = base / "notebooks" / "base" / "analise_exploratoria.ipynb"
notebook = nbformat.read(caminho, as_version=4)
ambiente = {"__name__": "__main__"}
numero = 0
# As figuras são incorporadas como imagens, da mesma forma que numa saída inline.
plt.show = lambda: None
for indice, celula in enumerate(notebook.cells):
    if celula.cell_type != "code":
        continue
    numero += 1
    if "--desde-divisao" in sys.argv and numero < 40:
        # Reconstituir apenas os dados necessários. A EDA anterior não mudou.
        if numero in [1, 10, 32, 33, 34, 35]:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(celula.source, f"notebook:celula_{numero}", "exec"), ambiente)
        continue
    print(f"Executando celula {numero}", flush=True)
    captura = io.StringIO()
    outputs = []
    erro = None
    resultado = None
    arvore = ast.parse(celula.source)
    ultima = None
    if arvore.body and isinstance(arvore.body[-1], ast.Expr):
        ultima = ast.Expression(arvore.body.pop().value)
    try:
        with contextlib.redirect_stdout(captura):
            exec(compile(arvore, f"notebook:celula_{numero}", "exec"), ambiente)
            if ultima is not None:
                resultado = eval(compile(ultima, f"notebook:celula_{numero}", "eval"), ambiente)
    except Exception:
        erro = traceback.format_exc()
    if captura.getvalue():
        outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captura.getvalue()))
    if resultado is not None:
        exibicao = {"text/plain": repr(resultado)}
        if hasattr(resultado, "_repr_html_"):
            html = resultado._repr_html_()
            if html:
                exibicao["text/html"] = html
        outputs.append(nbformat.v4.new_output("execute_result", data=exibicao, execution_count=numero))
    for figura in plt.get_fignums():
        imagem = io.BytesIO()
        plt.figure(figura).savefig(imagem, format="png", bbox_inches="tight", dpi=110)
        outputs.append(nbformat.v4.new_output("display_data", data={
            "image/png": base64.b64encode(imagem.getvalue()).decode("ascii")
        }))
    plt.close("all")
    if erro:
        outputs.append(nbformat.v4.new_output("error", ename="ExecutionError", evalue=erro, traceback=erro.splitlines()))
    celula.outputs = outputs
    celula.execution_count = numero
    nbformat.write(notebook, caminho)
    if erro:
        raise RuntimeError(erro)

nbformat.validate(notebook)
assert all(c.execution_count for c in notebook.cells if c.cell_type == "code")
assert not any(o.output_type == "error" for c in notebook.cells if c.cell_type == "code" for o in c.outputs)
for nome in ["treino", "validacao", "teste"]:
    frame = ambiente[{"treino": "X_treino", "validacao": "X_validacao", "teste": "X_teste"}[nome]]
    assert not frame.isna().any().any()
    assert ambiente["np"].isfinite(frame.to_numpy(dtype=float)).all()
assert ambiente["X_treino"].columns.equals(ambiente["X_validacao"].columns)
assert ambiente["X_treino"].columns.equals(ambiente["X_teste"].columns)
assert ambiente["treino"].equals(ambiente["pd"].read_csv(base / "input" / "train.csv"))
assert ambiente["teste"].equals(ambiente["pd"].read_csv(base / "input" / "test.csv"))
assert ambiente["y_treino"].equals(ambiente["treino"].loc[ambiente["X_treino"].index, "TARGET"])
assert ambiente["y_validacao"].equals(ambiente["treino"].loc[ambiente["X_validacao"].index, "TARGET"])
ambiente["medianas"].to_csv(base / "dados_modelagem" / "medianas_subtreino.csv", header=["mediana"])
ambiente["pd"].DataFrame({"coluna_removida_constante_subtreino": ambiente["constantes"]}).to_csv(
    base / "dados_modelagem" / "colunas_constantes.csv", index=False)
(base / "dados_modelagem" / "colunas_modelo.json").write_text(
    json.dumps(ambiente["X_treino"].columns.tolist(), indent=2), encoding="utf-8")
ambiente["pd"].DataFrame({"SK_ID_CURR": ambiente["treino"]["SK_ID_CURR"],
    "particao": ambiente["np"].where(ambiente["treino"].index.isin(ambiente["X_treino"].index),
                                    "treino", "validacao")}).to_csv(
    base / "dados_modelagem" / "particoes.csv", index=False)
resumo = {"n_treino": len(ambiente["X_treino"]), "n_validacao": len(ambiente["X_validacao"]),
    "n_teste": len(ambiente["X_teste"]), "preditores": len(ambiente["X_treino"].columns),
    "proporcao_treino": 0.8, "proporcao_validacao": 0.2, "random_state": 42,
    "taxa_alvo_treino": float(ambiente["y_treino"].mean()),
    "taxa_alvo_validacao": float(ambiente["y_validacao"].mean())}
(base / "dados_modelagem" / "preparacao.json").write_text(
    json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(resumo, ensure_ascii=False), flush=True)
print(f"Notebook validado: {numero} celulas executadas; originais preservados.", flush=True)
