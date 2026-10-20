"""Executa as etapas e salva o notebook com saídas, gráficos e relatório HTML."""
from pathlib import Path
import os
import re
import io
import json
import base64
import contextlib
import traceback
import markdown

base = Path(__file__).resolve().parent
os.chdir(base)
fonte = (base / "analise_dados.py").read_text(encoding="utf-8")
blocos = re.split(r"(?m)^# %%([^\n]*)\n", fonte)
celulas = []
ambiente = {"__name__": "__main__"}
numero = 0
for indice in range(1, len(blocos), 2):
    tipo = blocos[indice]
    conteudo = blocos[indice + 1].strip()
    if "markdown" in tipo:
        texto = "\n".join(re.sub(r"^# ?", "", linha) for linha in conteudo.splitlines())
        celulas.append({"cell_type": "markdown", "metadata": {}, "source": texto})
        print(texto.splitlines()[0], flush=True)
        continue
    numero += 1
    saida = io.StringIO()
    anteriores = {p.name: p.stat().st_mtime_ns for p in (base / "analise" / "graficos").glob("*.png")}
    erro = None
    try:
        with contextlib.redirect_stdout(saida):
            exec(compile(conteudo, f"analise_dados:celula_{numero}", "exec"), ambiente)
    except Exception:
        erro = traceback.format_exc()
    outputs = []
    if saida.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": saida.getvalue()})
        print(saida.getvalue(), flush=True)
    for imagem in sorted((base / "analise" / "graficos").glob("*.png")):
        if anteriores.get(imagem.name) != imagem.stat().st_mtime_ns:
            outputs.append({"output_type": "display_data", "metadata": {}, "data": {
                "image/png": base64.b64encode(imagem.read_bytes()).decode("ascii"),
                "text/plain": imagem.name}})
    if erro:
        outputs.append({"output_type": "stream", "name": "stderr", "text": erro})
    celulas.append({"cell_type": "code", "execution_count": numero, "metadata": {},
                    "source": conteudo, "outputs": outputs})
    notebook = {"cells": celulas, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    destino = base / "notebooks" / "referencia" / "analise_exploratoria_detalhada.ipynb"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    if erro:
        raise RuntimeError(erro)

texto_relatorio = (base / "analise" / "relatorio_analise.md").read_text(encoding="utf-8")
corpo = markdown.markdown(texto_relatorio, extensions=["tables", "fenced_code"])
pagina = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<title>Análise Home Credit GCI</title><style>
body{font-family:system-ui,sans-serif;max-width:1140px;margin:40px auto;padding:0 24px;color:#20313e;line-height:1.65}
h1,h2{line-height:1.25;color:#164d6a}h2{margin-top:2.2em}
table{display:block;overflow-x:auto;border-collapse:collapse;font-size:.87em;margin:24px 0}
th,td{padding:9px 12px;border-bottom:1px solid #dce4e9;text-align:right}th:first-child,td:first-child{text-align:left}
th{background:#edf3f6}img{max-width:100%;height:auto;border:1px solid #edf3f6}a{color:#17618b}
code{background:#f1f4f6;padding:2px 4px}@media print{body{max-width:none;margin:0}img{max-height:90vh;object-fit:contain}}
</style><body>''' + corpo + "</body></html>"
(base / "analise" / "relatorio_analise.html").write_text(pagina, encoding="utf-8")
print("Notebook executado e relatórios salvos.", flush=True)
