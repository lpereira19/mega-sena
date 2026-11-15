"""Atualização completa de um jogo com os concursos mais recentes.

Roda em sequência:

1. ETL: baixa da API os concursos que ainda não estão em ``data/raw`` e regera o CSV processado
2. Features por concurso
3. Testes de hipótese
4. Testes de distância entre as dezenas
5. Backtest das estratégias
6. Figuras em ``reports/figures/<jogo>``
7. Seção de resultados do README e reexecução do notebook (só na Mega-Sena, por enquanto)

Uso::

    python -m mega_sena.pipeline                  # atualização completa da Mega-Sena
    python -m mega_sena.pipeline --jogo lotofacil # escolhe o jogo
    python -m mega_sena.pipeline --jogo todos     # todos os jogos registrados
    python -m mega_sena.pipeline --skip-download  # só reprocessa os dados já baixados
"""

from pathlib import Path

from loguru import logger
from nbclient import NotebookClient
import nbformat
import pandas as pd
import typer

from mega_sena import dataset, distances, features, hypothesis, plots, report
from mega_sena.config import PROJ_ROOT, RANDOM_SEED
from mega_sena.games import GAMES, MEGA_SENA, GameSpec, get_game
from mega_sena.modeling import backtest

app = typer.Typer()

NOTEBOOK_FILE = PROJ_ROOT / "notebooks" / "1.0-lsp-analise-mega-sena.ipynb"


def _last_contest(spec: GameSpec) -> int:
    """Número do último concurso no CSV processado do jogo (0 se ainda não existir)."""
    path = spec.paths.processed
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


def run(
    spec: GameSpec,
    skip_download: bool = False,
    run_notebook: bool = True,
    n_sims: int = 2000,
    seed: int = RANDOM_SEED,
) -> int:
    """Atualiza dados, análises e figuras de um jogo.

    Returns:
        Número do último concurso processado.
    """
    before = _last_contest(spec)
    dataset.run(spec, skip_download=skip_download)
    after = _last_contest(spec)
    if after == before:
        logger.info(
            f"{spec.nome}: nenhum concurso novo (último: {after}); recalculando mesmo assim."
        )
    else:
        logger.info(f"{spec.nome}: concursos novos: {before + 1} a {after}.")

    features.run(spec)
    hypothesis.run(spec, n_sims, seed)
    distances.run(spec, n_sims, seed)
    backtest.run(spec, n_sims=n_sims, seed=seed)
    plots.run(spec)

    if spec == MEGA_SENA:
        report.run()
        if run_notebook:
            logger.info("Reexecutando o notebook...")
            execute_notebook()
            logger.info(f"Notebook atualizado: {NOTEBOOK_FILE}")

    logger.success(f"{spec.nome}: projeto atualizado até o concurso {after}.")
    return after


@app.command()
def main(
    jogo: str = typer.Option(MEGA_SENA.slug, help="Jogo (slug da API) ou 'todos'."),
    skip_download: bool = typer.Option(
        False, help="Não consulta a API; usa os dados já baixados."
    ),
    run_notebook: bool = typer.Option(True, help="Reexecuta o notebook no final."),
    n_sims: int = typer.Option(2000, help="Simulações de Monte Carlo (testes e backtest)."),
    seed: int = RANDOM_SEED,
):
    """Atualiza dados, análises, figuras, README e notebook."""
    specs = list(GAMES.values()) if jogo == "todos" else [get_game(jogo)]
    for spec in specs:
        run(spec, skip_download, run_notebook, n_sims, seed)


if __name__ == "__main__":
    app()
