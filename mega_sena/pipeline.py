"""Atualização completa do projeto com os concursos mais recentes.

Roda em sequência:

1. ETL: baixa da API os concursos que ainda não estão em ``data/raw`` e regera o CSV processado
2. Features por concurso
3. Testes de hipótese
4. Backtest das estratégias
5. Figuras em ``reports/figures``
6. Seção de resultados do README
7. Reexecução do notebook, para as saídas salvas refletirem os novos dados

Uso::

    python -m mega_sena.pipeline                  # atualização completa
    python -m mega_sena.pipeline --skip-download  # só reprocessa os dados já baixados
"""

from pathlib import Path

from loguru import logger
from nbclient import NotebookClient
import nbformat
import pandas as pd
import typer

from mega_sena import dataset, features, hypothesis, plots, report
from mega_sena.config import PROCESSED_RESULTS_FILE, PROJ_ROOT, RANDOM_SEED
from mega_sena.modeling import backtest

app = typer.Typer()

NOTEBOOK_FILE = PROJ_ROOT / "notebooks" / "1.0-lsp-analise-mega-sena.ipynb"


def _last_contest(path: Path = PROCESSED_RESULTS_FILE) -> int:
    """Número do último concurso no CSV processado (0 se ainda não existir)."""
    return int(pd.read_csv(path, usecols=["concurso"])["concurso"].max()) if path.exists() else 0


def execute_notebook(path: Path = NOTEBOOK_FILE, timeout: int = 600) -> None:
    """Reexecuta o notebook e salva as saídas no próprio arquivo."""
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    nbformat.write(notebook, path)


@app.command()
def main(
    skip_download: bool = typer.Option(
        False, help="Não consulta a API; usa os dados já baixados."
    ),
    run_notebook: bool = typer.Option(True, help="Reexecuta o notebook no final."),
    n_sims: int = typer.Option(2000, help="Simulações de Monte Carlo (testes e backtest)."),
    seed: int = RANDOM_SEED,
):
    """Atualiza dados, análises, figuras, README e notebook."""
    before = _last_contest()

    dataset.main(
        raw_path=dataset.RAW_RESULTS_FILE,
        output_path=PROCESSED_RESULTS_FILE,
        skip_download=skip_download,
        workers=2,
        delay=0.25,
    )
    after = _last_contest()
    if after == before:
        logger.info(f"Nenhum concurso novo (último: {after}); recalculando mesmo assim.")
    else:
        logger.info(f"Concursos novos: {before + 1} a {after}.")

    features.main(input_path=PROCESSED_RESULTS_FILE, output_path=features.FEATURES_FILE)
    hypothesis.main(
        input_path=PROCESSED_RESULTS_FILE,
        output_path=hypothesis.HYPOTHESIS_TESTS_FILE,
        n_sims=n_sims,
        seed=seed,
    )
    backtest.main(
        input_path=PROCESSED_RESULTS_FILE,
        summary_path=backtest.BACKTEST_SUMMARY_FILE,
        hits_path=backtest.BACKTEST_HITS_FILE,
        null_path=backtest.BACKTEST_NULL_FILE,
        start=100,
        n_sims=n_sims,
        seed=seed,
    )
    plots.main(input_path=PROCESSED_RESULTS_FILE, output_dir=plots.FIGURES_DIR)
    report.main(
        input_path=PROCESSED_RESULTS_FILE,
        tests_path=report.HYPOTHESIS_TESTS_FILE,
        summary_path=report.BACKTEST_SUMMARY_FILE,
        readme_path=report.README_FILE,
    )
    if run_notebook:
        logger.info("Reexecutando o notebook...")
        execute_notebook()
        logger.info(f"Notebook atualizado: {NOTEBOOK_FILE}")

    logger.success(f"Projeto atualizado até o concurso {after}.")


if __name__ == "__main__":
    app()
