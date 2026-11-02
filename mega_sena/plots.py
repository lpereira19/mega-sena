"""Visualizações da análise exploratória e do backtest.

Cada função recebe os dados e devolve uma ``matplotlib.figure.Figure``, então funciona tanto
num notebook quanto no comando abaixo, que salva todas as figuras em ``reports/figures``::

    python -m mega_sena.plots

Em todos os gráficos, a referência teórica (o que se espera de um sorteio uniforme) aparece em
cinza, para o desvio observado ficar visível.
"""

from pathlib import Path

from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from mega_sena.config import (
    BACKTEST_HITS_FILE,
    BACKTEST_NULL_FILE,
    BACKTEST_SUMMARY_FILE,
    FIGURES_DIR,
    NUMBERS_PER_DRAW,
    PROCESSED_RESULTS_FILE,
    TOTAL_NUMBERS,
)
from mega_sena.dataset import load_processed
from mega_sena.features import (
    N_QUADRANTS,
    QUADRANT_COLUMNS,
    VOLANTE_COLS,
    VOLANTE_ROWS,
    build_features,
    cumulative_frequencies,
    draws_matrix,
    number_frequencies,
)
from mega_sena.hypothesis import hypergeom_pmf
from mega_sena.modeling.backtest import EXPECTED_HITS

app = typer.Typer()

# Paleta categórica validada para daltonismo (ordem fixa: cada série mantém sua cor).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
REFERENCE = "#8a8984"  # referência teórica / esperado
MUTED = "#c9c8c2"  # séries de contexto
TEXT, TEXT_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

P_NUMBER = NUMBERS_PER_DRAW / TOTAL_NUMBERS  # probabilidade de uma dezena sair num concurso


def set_style() -> None:
    """Tema visual comum a todas as figuras: grade discreta, sem bordas desnecessárias."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#ebeae6",
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def plot_number_frequency(df: pd.DataFrame, n_highlight: int = 6) -> Figure:
    """Frequência acumulada de cada dezena, com a faixa esperada sob sorteio uniforme.

    A faixa cinza é o intervalo de 95% da Binomial(n_concursos, 0,1) de cada dezena. Com 60
    dezenas, espera-se que cerca de 3 fiquem fora dela por puro acaso.
    """
    freq = number_frequencies(draws_matrix(df))
    n = len(df)
    expected = n * P_NUMBER
    half_band = 1.96 * np.sqrt(n * P_NUMBER * (1 - P_NUMBER))
    top = set(freq.nlargest(n_highlight).index)
    bottom = set(freq.nsmallest(n_highlight).index)
    colors = [SERIES[1] if d in top else SERIES[0] if d in bottom else MUTED for d in freq.index]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhspan(expected - half_band, expected + half_band, color=REFERENCE, alpha=0.15, lw=0)
    ax.bar(freq.index, freq.values, color=colors, width=0.8, edgecolor=SURFACE, linewidth=1)
    ax.axhline(expected, color=REFERENCE, lw=1.5, ls="--")
    ax.annotate(
        f"esperado: {expected:.0f} (faixa de 95%)",
        xy=(TOTAL_NUMBERS + 0.6, expected),
        va="center",
        color=TEXT_SECONDARY,
        fontsize=9,
        annotation_clip=False,
    )
    for d in top | bottom:
        ax.annotate(
            str(freq[d]),
            (d, freq[d]),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
            color=TEXT_SECONDARY,
        )

    ax.set_xticks(range(1, TOTAL_NUMBERS + 1))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(0.3, TOTAL_NUMBERS + 0.7)
    ax.set_ylim(freq.min() * 0.85, freq.max() * 1.12)  # folga para a legenda
    ax.set_xlabel("Dezena")
    ax.set_ylabel("Vezes sorteada")
    ax.set_title(f"Frequência de cada dezena em {n} concursos")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=SERIES[1]),
            plt.Rectangle((0, 0), 1, 1, color=SERIES[0]),
            plt.Rectangle((0, 0), 1, 1, color=MUTED),
        ],
        labels=[f"{n_highlight} mais sorteadas", f"{n_highlight} menos sorteadas", "demais"],
        loc="upper left",
        ncols=3,
    )
    return fig


def _spread_labels(values: pd.Series, min_gap: float) -> pd.Series:
    """Afasta posições verticais de rótulos para que fiquem a pelo menos ``min_gap`` entre si."""
    ordered = values.sort_values()
    positions = ordered.to_numpy(dtype=float, copy=True)
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1] + min_gap)
    return pd.Series(positions, index=ordered.index)


def plot_cumulative_frequency(df: pd.DataFrame, n_highlight: int = 3) -> Figure:
    """Evolução do desvio da frequência acumulada de cada dezena em relação ao esperado.

    Cada linha é ``frequência acumulada - esperado`` ao longo dos concursos. Sob aleatoriedade,
    cada linha é um passeio aleatório; o envelope de 95% cresce com a raiz do número de
    concursos. As dezenas "quentes" de hoje são simplesmente as que estão no topo desse
    passeio agora.
    """
    cumulative = cumulative_frequencies(draws_matrix(df))
    t = np.arange(1, len(cumulative) + 1)
    deviation = cumulative.sub(t * P_NUMBER, axis=0)
    deviation.index = df["concurso"]
    final = deviation.iloc[-1]
    highlighted = [*final.nlargest(n_highlight).index, *final.nsmallest(n_highlight).index]
    envelope = 1.96 * np.sqrt(t * P_NUMBER * (1 - P_NUMBER))

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.fill_between(
        deviation.index,
        -envelope,
        envelope,
        color=REFERENCE,
        alpha=0.15,
        lw=0,
        label="envelope de 95% (uma dezena)",
    )
    for number in deviation.columns.difference(highlighted):
        ax.plot(deviation.index, deviation[number], color=MUTED, lw=0.6, alpha=0.6)
    label_y = _spread_labels(final[highlighted], min_gap=0.035 * np.ptp(deviation.to_numpy()))
    for i, number in enumerate(highlighted):
        color = SERIES[1] if i < n_highlight else SERIES[0]
        ax.plot(deviation.index, deviation[number], color=color, lw=2)
        ax.annotate(
            f"{number:02d}",
            (deviation.index[-1], label_y[number]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT,
        )
    ax.axhline(0, color=REFERENCE, lw=1.5, ls="--")

    ax.set_xlabel("Concurso")
    ax.set_ylabel("Frequência acumulada − esperado")
    ax.set_title("Desvio acumulado de cada dezena em relação ao esperado")
    ax.legend(
        handles=[
            plt.Line2D([], [], color=SERIES[1], lw=2),
            plt.Line2D([], [], color=SERIES[0], lw=2),
            plt.Line2D([], [], color=MUTED, lw=1),
            plt.Rectangle((0, 0), 1, 1, color=REFERENCE, alpha=0.15),
        ],
        labels=[
            f"{n_highlight} mais sorteadas hoje",
            f"{n_highlight} menos sorteadas hoje",
            "demais dezenas",
            "envelope de 95%",
        ],
        loc="upper left",
        ncols=4,
    )
    return fig


def _observed_vs_expected_bars(ax, observed: np.ndarray, expected: np.ndarray, labels) -> None:
    """Barras lado a lado: proporção observada (cor) vs. esperada (cinza)."""
    x = np.arange(len(observed))
    width = 0.38
    ax.bar(
        x - width / 2,
        observed,
        width,
        color=SERIES[0],
        label="observado",
        edgecolor=SURFACE,
        linewidth=1,
    )
    ax.bar(
        x + width / 2,
        expected,
        width,
        color=REFERENCE,
        alpha=0.6,
        label="esperado (hipergeométrica)",
        edgecolor=SURFACE,
        linewidth=1,
    )
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right")


def plot_even_odd(df: pd.DataFrame) -> Figure:
    """Distribuição de pares e ímpares por concurso vs. a hipergeométrica (30 pares em 60)."""
    evens = build_features(df)["pares"]
    observed = np.bincount(evens, minlength=NUMBERS_PER_DRAW + 1) / len(evens)
    expected = hypergeom_pmf(TOTAL_NUMBERS // 2)
    labels = [f"{k}P / {NUMBERS_PER_DRAW - k}I" for k in range(NUMBERS_PER_DRAW + 1)]

    fig, ax = plt.subplots(figsize=(10, 5))
    _observed_vs_expected_bars(ax, observed, expected, labels)
    for x, value in enumerate(observed):
        ax.annotate(
            f"{value:.1%}",
            (x - 0.19, value),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
            color=TEXT_SECONDARY,
        )
    ax.set_xlabel("Composição do concurso (P = pares, I = ímpares)")
    ax.set_ylabel("Proporção dos concursos")
    ax.set_title("Pares vs. ímpares por concurso")
    return fig


def plot_quadrants(df: pd.DataFrame) -> Figure:
    """Quadrantes do volante: mapa de calor das frequências e distribuição por concurso.

    À esquerda, o volante 6x10 colorido pela frequência de cada dezena, com os quadrantes
    delimitados. À direita, a proporção de concursos com ``k`` dezenas em cada quadrante,
    comparada à hipergeométrica (15 dezenas por quadrante).
    """
    freq = number_frequencies(draws_matrix(df))
    grid = freq.to_numpy().reshape(VOLANTE_ROWS, VOLANTE_COLS)
    features = build_features(df)

    fig, (ax_map, ax_dist) = plt.subplots(
        1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    sns.heatmap(
        grid,
        ax=ax_map,
        cmap=SEQUENTIAL,
        linewidths=2,
        linecolor=SURFACE,
        square=True,
        annot=np.arange(1, TOTAL_NUMBERS + 1).reshape(grid.shape),
        fmt="02d",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Vezes sorteada", "shrink": 0.8},
        xticklabels=False,
        yticklabels=False,
    )
    ax_map.axhline(VOLANTE_ROWS / 2, color=TEXT, lw=2.5)
    ax_map.axvline(VOLANTE_COLS / 2, color=TEXT, lw=2.5)
    for q, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)], start=1):
        total = features[f"q{q}"].sum()
        ax_map.text(
            col * VOLANTE_COLS / 2 + 0.1,
            row * VOLANTE_ROWS / 2 + 0.05,
            f"Q{q}: {total} ({total / (len(df) * NUMBERS_PER_DRAW):.1%})",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=TEXT,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
    ax_map.set_title("Volante: frequência por dezena e total por quadrante")

    k = np.arange(NUMBERS_PER_DRAW + 1)
    expected = hypergeom_pmf(TOTAL_NUMBERS // N_QUADRANTS)
    ax_dist.bar(
        k, expected, color=REFERENCE, alpha=0.35, width=0.8, label="esperado (hipergeométrica)"
    )
    for i, column in enumerate(QUADRANT_COLUMNS):
        observed = np.bincount(features[column], minlength=NUMBERS_PER_DRAW + 1) / len(df)
        ax_dist.plot(
            k,
            observed,
            marker="o",
            ms=6,
            lw=2,
            color=SERIES[i],
            label=column.upper(),
            mec=SURFACE,
            mew=1.5,
        )
    ax_dist.set_xticks(k)
    ax_dist.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_dist.set_xlabel("Dezenas sorteadas no quadrante")
    ax_dist.set_ylabel("Proporção dos concursos")
    ax_dist.set_title("Dezenas por quadrante em cada concurso")
    ax_dist.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_backtest(hits: pd.DataFrame, null_means: np.ndarray, summary: pd.DataFrame) -> Figure:
    """Resultado do backtest: distribuição de acertos e médias contra a distribuição nula.

    À esquerda, a proporção de concursos com 0, 1, 2 e 3+ acertos por estratégia, comparada à
    hipergeométrica (60, 6, 6). À direita, o histograma das médias de acertos de milhares de
    apostadores aleatórios, com a média de cada estratégia marcada.
    """
    strategies = summary["estrategia"].tolist()
    colors = dict(zip(strategies, SERIES))
    max_bin = 3  # 3 ou mais acertos ficam juntos: quadras, quinas e senas são raríssimas
    labels = ["0", "1", "2", f"{max_bin}+"]
    expected = hypergeom_pmf(NUMBERS_PER_DRAW)
    expected = np.append(expected[:max_bin], expected[max_bin:].sum())

    fig, (ax_dist, ax_null) = plt.subplots(1, 2, figsize=(16, 5.5))
    x = np.arange(len(labels))
    width = 0.8 / (len(strategies) + 1)
    ax_dist.bar(
        x - 0.4 + width / 2,
        expected,
        width,
        color=REFERENCE,
        alpha=0.6,
        label="esperado (acaso)",
        edgecolor=SURFACE,
        linewidth=1,
    )
    for i, name in enumerate(strategies, start=1):
        k = np.bincount(
            hits.loc[hits["estrategia"] == name, "acertos"], minlength=NUMBERS_PER_DRAW + 1
        )
        observed = np.append(k[:max_bin], k[max_bin:].sum()) / k.sum()
        ax_dist.bar(
            x - 0.4 + width * (i + 0.5),
            observed,
            width,
            color=colors[name],
            label=name,
            edgecolor=SURFACE,
            linewidth=1,
        )
    ax_dist.set_xticks(x, labels)
    ax_dist.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_dist.set_xlabel("Acertos por concurso")
    ax_dist.set_ylabel("Proporção das apostas")
    ax_dist.set_title("Distribuição de acertos por estratégia")
    ax_dist.legend(loc="upper right")

    ax_null.hist(
        null_means,
        bins=40,
        color=MUTED,
        edgecolor=SURFACE,
        label=f"{len(null_means)} apostadores aleatórios",
    )
    ax_null.axvline(
        EXPECTED_HITS, color=REFERENCE, lw=1.5, ls="--", label=f"esperado: {EXPECTED_HITS:.1f}"
    )
    for _, row in summary.iterrows():
        ax_null.axvline(
            row["media_acertos"],
            color=colors[row["estrategia"]],
            lw=2.5,
            label=f"{row['estrategia']}: {row['media_acertos']:.3f} "
            f"(p = {row['p_valor_monte_carlo']:.3f}; "
            f"Bonferroni = {row['p_valor_mc_bonferroni']:.2f})",
        )
    ax_null.set_xlabel("Média de acertos por concurso")
    ax_null.set_ylabel("Apostadores simulados")
    ax_null.set_title("Média de acertos vs. distribuição nula")
    ax_null.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2, fontsize=8.5)
    fig.tight_layout()
    return fig


@app.command()
def main(
    input_path: Path = PROCESSED_RESULTS_FILE,
    output_dir: Path = FIGURES_DIR,
):
    """Gera e salva todas as figuras. A do backtest só é gerada se o backtest já tiver rodado."""
    set_style()
    df = load_processed(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "frequencia_dezenas.png": plot_number_frequency(df),
        "frequencia_acumulada.png": plot_cumulative_frequency(df),
        "pares_impares.png": plot_even_odd(df),
        "quadrantes.png": plot_quadrants(df),
    }
    if BACKTEST_SUMMARY_FILE.exists():
        figures["backtest.png"] = plot_backtest(
            pd.read_csv(BACKTEST_HITS_FILE),
            pd.read_csv(BACKTEST_NULL_FILE)["media_acertos"].to_numpy(),
            pd.read_csv(BACKTEST_SUMMARY_FILE),
        )
    else:
        logger.warning("Backtest ainda não rodou; pulando backtest.png.")

    for name, fig in figures.items():
        fig.savefig(output_dir / name)
        plt.close(fig)
        logger.info(f"Figura salva: {output_dir / name}")
    logger.success(f"{len(figures)} figuras geradas.")


if __name__ == "__main__":
    app()
