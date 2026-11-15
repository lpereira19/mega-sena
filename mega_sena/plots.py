"""Visualizações da análise exploratória e do backtest.

Cada função recebe os dados e devolve uma ``matplotlib.figure.Figure``, então funciona tanto
num notebook quanto no comando abaixo, que salva todas as figuras em
``reports/figures/<jogo>``::

    python -m mega_sena.plots

Em todos os gráficos, a referência teórica (o que se espera de um sorteio uniforme) aparece em
cinza, para o desvio observado ficar visível.
"""

from collections.abc import Callable

from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from mega_sena import distances
from mega_sena.dataset import MONTHS, load_processed
from mega_sena.features import (
    QUADRANT_COLUMNS,
    build_features,
    cumulative_frequencies,
    draws_matrix,
    even_count,
    number_frequencies,
    quadrant_sizes,
)
from mega_sena.games import MEGA_SENA, GameSpec, get_game
from mega_sena.hypothesis import extra_values, hypergeom_pmf
from mega_sena.modeling.backtest import expected_hits, hits_pmf

app = typer.Typer()

# Paleta categórica validada para daltonismo (ordem fixa: cada série mantém sua cor).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
REFERENCE = "#8a8984"  # referência teórica / esperado
MUTED = "#c9c8c2"  # séries de contexto
TEXT, TEXT_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def _percent(value: float, _position=None) -> str:
    """Rótulo de eixo em porcentagem, com uma casa decimal só quando a marca precisa (12,5%)."""
    return f"{value * 100:.1f}".removesuffix(".0") + "%"


PERCENT = plt.FuncFormatter(_percent)


def _p_number(spec: GameSpec) -> float:
    """Probabilidade de uma dezena sair num concurso: K/N."""
    return spec.sorteadas / spec.n


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


def plot_number_frequency(
    df: pd.DataFrame, n_highlight: int = 6, spec: GameSpec = MEGA_SENA
) -> Figure:
    """Frequência acumulada de cada dezena, com a faixa esperada sob sorteio uniforme.

    A faixa cinza é o intervalo de 95% da Binomial(n_concursos, K/N) de cada dezena. Com N
    dezenas, espera-se que cerca de 5% delas fiquem fora da faixa por puro acaso.
    """
    freq = number_frequencies(draws_matrix(df), spec)
    n = len(df)
    p = _p_number(spec)
    expected = n * p
    half_band = 1.96 * np.sqrt(n * p * (1 - p))
    top = set(freq.nlargest(n_highlight).index)
    bottom = set(freq.nsmallest(n_highlight).index)
    colors = [SERIES[1] if d in top else SERIES[0] if d in bottom else MUTED for d in freq.index]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhspan(expected - half_band, expected + half_band, color=REFERENCE, alpha=0.15, lw=0)
    ax.bar(freq.index, freq.values, color=colors, width=0.8, edgecolor=SURFACE, linewidth=1)
    ax.axhline(expected, color=REFERENCE, lw=1.5, ls="--")
    ax.annotate(
        f"esperado: {expected:.0f} (faixa de 95%)",
        xy=(spec.maior + 0.6, expected),
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

    ax.set_xticks(spec.numbers)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(spec.menor - 0.7, spec.maior + 0.7)
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


def plot_cumulative_frequency(
    df: pd.DataFrame, n_highlight: int = 3, spec: GameSpec = MEGA_SENA
) -> Figure:
    """Evolução do desvio da frequência acumulada de cada dezena em relação ao esperado.

    Cada linha é ``frequência acumulada - esperado`` ao longo dos concursos. Sob aleatoriedade,
    cada linha é um passeio aleatório; o envelope de 95% cresce com a raiz do número de
    concursos. As dezenas "quentes" de hoje são simplesmente as que estão no topo desse
    passeio agora.
    """
    p = _p_number(spec)
    cumulative = cumulative_frequencies(draws_matrix(df), spec)
    t = np.arange(1, len(cumulative) + 1)
    deviation = cumulative.sub(t * p, axis=0)
    deviation.index = df["concurso"]
    final = deviation.iloc[-1]
    highlighted = [*final.nlargest(n_highlight).index, *final.nsmallest(n_highlight).index]
    envelope = 1.96 * np.sqrt(t * p * (1 - p))

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


def _observed_vs_expected_bars(
    ax,
    observed: np.ndarray,
    expected: np.ndarray,
    labels,
    expected_label: str = "esperado (hipergeométrica)",
    annotate: bool = False,
) -> None:
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
        label=expected_label,
        edgecolor=SURFACE,
        linewidth=1,
    )
    if annotate:
        for i, value in enumerate(observed):
            ax.annotate(
                f"{value:.1%}",
                (i - width / 2, value),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
                color=TEXT_SECONDARY,
            )
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(PERCENT)
    ax.legend(loc="upper right")


def _tail_binned(
    observed: np.ndarray,
    pmf: np.ndarray,
    start: int,
    label: Callable[[int, str], str] = lambda v, plus: f"{v}{plus}",
    tail: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Agrupa as caudas raras de uma distribuição discreta para o gráfico de barras.

    Os valores são ``start, start + 1, ...``. A cauda de cima começa no primeiro valor ``v``
    com P(X >= v) < ``tail`` e vira a categoria "v+"; a de baixo junta os valores com
    P(X <= v) < ``tail`` em "≤v". Assim as categorias se ajustam a cada jogo (na Mega-Sena,
    "3+" pares consecutivos; na Lotofácil, em torno de 8).

    Args:
        observed: Proporção observada de cada valor (mesmo tamanho de ``pmf``).
        pmf: Probabilidade teórica de cada valor.
        start: Valor da primeira posição.
        label: Rótulo de um valor; recebe o valor e ``"+"`` na categoria da cauda de cima.
        tail: Probabilidade máxima de cada cauda agrupada.

    Returns:
        Observado e esperado já agrupados, e os rótulos das categorias.
    """
    pmf = np.asarray(pmf, dtype=float)
    observed = np.asarray(observed, dtype=float)
    upper = np.flatnonzero(np.cumsum(pmf[::-1])[::-1] < tail)
    hi = int(upper[0]) if len(upper) else len(pmf) - 1
    lower = np.flatnonzero(np.cumsum(pmf) < tail)
    lo = int(lower[-1]) if len(lower) else -1

    def group(values: np.ndarray) -> np.ndarray:
        head = [values[: lo + 1].sum()] if lo >= 0 else []
        return np.array([*head, *values[lo + 1 : hi], values[hi:].sum()])

    labels = [
        *([label(start, "")] if lo == 0 else []),  # cauda de baixo com um valor só
        *([f"≤{label(start + lo, '')}"] if lo > 0 else []),
        *(label(start + i, "") for i in range(lo + 1, hi)),
        label(start + hi, "+" if hi < len(pmf) - 1 else ""),
    ]
    return group(observed), group(pmf), labels


def plot_even_odd(df: pd.DataFrame, spec: GameSpec = MEGA_SENA) -> Figure:
    """Distribuição de pares e ímpares por concurso vs. a hipergeométrica."""
    k = spec.sorteadas
    evens = build_features(df, spec)["pares"]
    observed = np.bincount(evens, minlength=k + 1) / len(evens)
    expected = hypergeom_pmf(even_count(spec), spec)
    labels = [f"{j}P / {k - j}I" for j in range(k + 1)]

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


def plot_quadrants(df: pd.DataFrame, spec: GameSpec = MEGA_SENA) -> Figure:
    """Quadrantes do volante: mapa de calor das frequências e distribuição por concurso.

    À esquerda, o volante colorido pela frequência de cada dezena, com os quadrantes
    delimitados. À direita, a proporção de concursos com ``k`` dezenas em cada quadrante,
    comparada à hipergeométrica.
    """
    rows, cols = spec.volante
    split_row, split_col = rows // 2, cols // 2
    k_max = spec.sorteadas
    freq = number_frequencies(draws_matrix(df), spec)
    # A grade pode ter a última linha incompleta (Dia de Sorte): as células sobrando ficam vazias.
    grid = np.full(rows * cols, np.nan)
    grid[: spec.n] = freq.to_numpy()
    grid = grid.reshape(rows, cols)
    labels = np.full(rows * cols, "", dtype=object)
    labels[: spec.n] = [f"{number:02d}" for number in spec.numbers]
    features = build_features(df, spec)
    sizes = quadrant_sizes(spec)

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
        annot=labels.reshape(grid.shape),
        fmt="",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Vezes sorteada", "shrink": 0.8},
        xticklabels=False,
        yticklabels=False,
    )
    ax_map.axhline(split_row, color=TEXT, lw=2.5)
    ax_map.axvline(split_col, color=TEXT, lw=2.5)
    for q, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)], start=1):
        total = features[f"q{q}"].sum()
        ax_map.text(
            col * split_col + 0.1,
            row * split_row + 0.05,
            f"Q{q}: {total} ({total / (len(df) * k_max):.1%})",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=TEXT,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
    ax_map.set_title("Volante: frequência por dezena e total por quadrante")

    k = np.arange(k_max + 1)
    if len(set(sizes)) == 1:
        ax_dist.bar(
            k,
            hypergeom_pmf(sizes[0], spec),
            color=REFERENCE,
            alpha=0.35,
            width=0.8,
            label="esperado (hipergeométrica)",
        )
    for i, column in enumerate(QUADRANT_COLUMNS):
        observed = np.bincount(features[column], minlength=k_max + 1) / len(df)
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
        if len(set(sizes)) > 1:  # quadrantes de tamanhos diferentes: um esperado para cada
            ax_dist.plot(k, hypergeom_pmf(sizes[i], spec), color=SERIES[i], lw=1, ls="--")
    ax_dist.set_xticks(k)
    ax_dist.yaxis.set_major_formatter(PERCENT)
    ax_dist.set_xlabel("Dezenas sorteadas no quadrante")
    ax_dist.set_ylabel("Proporção dos concursos")
    ax_dist.set_title("Dezenas por quadrante em cada concurso")
    ax_dist.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _hit_bins(pmf: np.ndarray, min_share: float = 0.005) -> tuple[int, int]:
    """Faixa de acertos mostrada no gráfico: os valores com ao menos ``min_share`` de chance.

    As caudas raras são somadas nas pontas (ex.: "3+" na Mega-Sena).
    """
    frequent = np.flatnonzero(pmf >= min_share)
    return int(frequent.min()), int(frequent.max())


def plot_backtest(
    hits: pd.DataFrame,
    null_means: np.ndarray,
    summary: pd.DataFrame,
    spec: GameSpec = MEGA_SENA,
) -> Figure:
    """Resultado do backtest: distribuição de acertos e médias contra a distribuição nula.

    À esquerda, a proporção de concursos por número de acertos e estratégia, comparada à
    hipergeométrica (as caudas raras ficam agrupadas). À direita, o histograma das médias de
    acertos de milhares de apostadores aleatórios, com a média de cada estratégia marcada.
    """
    strategies = summary["estrategia"].tolist()
    colors = dict(zip(strategies, SERIES))
    pmf = hits_pmf(spec)
    lo, hi = _hit_bins(pmf)
    labels = [
        *([f"≤{lo}"] if lo > 0 else ["0"]),
        *map(str, range(lo + 1, hi)),
        f"{hi}+",
    ]

    def binned(values: np.ndarray) -> np.ndarray:
        return np.concatenate([[values[: lo + 1].sum()], values[lo + 1 : hi], [values[hi:].sum()]])

    expected = binned(pmf)
    expected_mean = expected_hits(spec)

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
            hits.loc[hits["estrategia"] == name, "acertos"], minlength=spec.sorteadas + 1
        )
        observed = binned(k) / k.sum()
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
    ax_dist.yaxis.set_major_formatter(PERCENT)
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
        expected_mean, color=REFERENCE, lw=1.5, ls="--", label=f"esperado: {expected_mean:.1f}"
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


def plot_distances(df: pd.DataFrame, spec: GameSpec = MEGA_SENA) -> Figure:
    """Quão próximas saem as dezenas de um concurso, comparado ao acaso.

    Quatro painéis: pares de dezenas consecutivas por concurso, maior sequência de
    consecutivas, menor distância entre duas dezenas e saltos entre dezenas vizinhas na ordem
    crescente. Em cinza, a distribuição exata sob sorteio uniforme.
    """
    draws = draws_matrix(df)
    n = len(draws)
    k = spec.sorteadas
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_pairs, ax_runs, ax_min, ax_gaps = axes.ravel()
    reference = "esperado (acaso)"

    pairs_pmf = distances.consecutive_pairs_pmf(spec)
    pairs = np.bincount(distances.consecutive_pairs(draws), minlength=len(pairs_pmf)) / n
    _observed_vs_expected_bars(
        ax_pairs, *_tail_binned(pairs, pairs_pmf, start=0), reference, annotate=True
    )
    ax_pairs.set_xlabel("Pares de dezenas consecutivas no concurso (ex.: 22 e 23)")
    ax_pairs.set_ylabel("Proporção dos concursos")
    ax_pairs.set_title("Quantos pares consecutivos saem por concurso")

    runs = np.bincount(distances.longest_run(draws), minlength=k + 1)[1:] / n
    runs_pmf = distances.longest_run_pmf(spec)
    runs_binned = _tail_binned(
        runs,
        runs_pmf,
        start=1,
        label=lambda v, plus: "nenhuma" if v == 1 and not plus else f"{v}{plus} seguidas",
    )
    runs_xlabel = "Maior sequência de consecutivas no concurso"
    if len(runs_binned[2]) > 6:  # muitas categorias (Lotofácil): rótulos curtos
        runs_binned = _tail_binned(runs, runs_pmf, start=1)
        runs_xlabel += " (1 = nenhuma)"
    _observed_vs_expected_bars(ax_runs, *runs_binned, reference, annotate=True)
    ax_runs.set_xlabel(runs_xlabel)
    ax_runs.set_ylabel("Proporção dos concursos")
    ax_runs.set_title("Maior sequência de dezenas seguidas")

    min_pmf = distances.min_gap_pmf(spec)
    smallest = np.bincount(distances.min_gap(draws), minlength=len(min_pmf) + 1)[1:] / n
    _observed_vs_expected_bars(
        ax_min, *_tail_binned(smallest, min_pmf, start=1), reference, annotate=True
    )
    ax_min.set_xlabel("Menor distância entre duas dezenas do concurso")
    ax_min.set_ylabel("Proporção dos concursos")
    ax_min.set_title("Quão perto ficam as duas dezenas mais próximas")

    gap_cap = spec.teto_salto
    gaps = np.minimum(distances.sorted_gaps(draws), gap_cap).ravel()
    observed = np.bincount(gaps, minlength=gap_cap + 1)[1:] / gaps.size
    x = np.arange(1, gap_cap + 1)
    ticks = list(range(1, gap_cap + 1)) if gap_cap <= 10 else [1, *range(5, gap_cap, 5), gap_cap]
    ax_gaps.bar(x, observed, color=SERIES[0], width=0.8, edgecolor=SURFACE, label="observado")
    ax_gaps.plot(
        x, distances.gap_pmf(spec), color=TEXT_SECONDARY, lw=2, marker="o", ms=4, label=reference
    )
    ax_gaps.set_xticks(ticks)
    ax_gaps.set_xticklabels([*map(str, ticks[:-1]), f"{gap_cap}+"])
    ax_gaps.yaxis.set_major_formatter(PERCENT)
    ax_gaps.set_xlabel("Salto entre duas dezenas vizinhas (ordem crescente)")
    ax_gaps.set_ylabel("Proporção dos saltos")
    ax_gaps.set_title("Saltos entre dezenas vizinhas")
    ax_gaps.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_neighborhood(df: pd.DataFrame, spec: GameSpec = MEGA_SENA) -> Figure:
    """Distância entre todas as dezenas de cada concurso e vizinha mais próxima de cada dezena.

    À esquerda, a distância entre cada par de dezenas de cada concurso: distâncias curtas são
    mais comuns só porque existem mais pares assim no volante (linha cinza). À direita, para
    cada dezena x, a distância média até a dezena mais próxima nos concursos em que x saiu, com
    a faixa de 95% esperada. As pontas do volante têm vizinhas só de um lado e, por isso, mais
    distantes.
    """
    draws = draws_matrix(df)
    fig, (ax_pairs, ax_near) = plt.subplots(1, 2, figsize=(16, 5.5))

    pairs = distances.pair_distances(draws).ravel()
    observed = np.bincount(pairs, minlength=spec.n)[1:] / pairs.size
    d = np.arange(1, spec.n)
    ax_pairs.bar(d, observed, color=SERIES[0], width=0.8, edgecolor=SURFACE, label="observado")
    ax_pairs.plot(
        d, distances.pair_distance_pmf(spec), color=TEXT_SECONDARY, lw=2, label="esperado (acaso)"
    )
    ax_pairs.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax_pairs.set_xlabel("Distância entre duas dezenas do mesmo concurso")
    ax_pairs.set_ylabel("Proporção dos pares")
    ax_pairs.set_title("Distância entre todos os pares de dezenas")
    ax_pairs.legend(loc="upper right")

    table = distances.nearest_by_number(draws, spec)
    numbers = table.index.to_numpy()
    half_band = 1.96 * table["desvio_padrao"] / np.sqrt(table["vezes"])
    colors = [SERIES[1] if z > 2 else SERIES[0] if z < -2 else MUTED for z in table["z"].fillna(0)]
    ax_near.fill_between(
        numbers,
        table["esperado"] - half_band,
        table["esperado"] + half_band,
        color=REFERENCE,
        alpha=0.2,
        lw=0,
        step="mid",
    )
    ax_near.bar(numbers, table["media"], color=colors, width=0.8, edgecolor=SURFACE)
    ax_near.plot(numbers, table["esperado"], color=TEXT_SECONDARY, lw=1.5, drawstyle="steps-mid")
    for number in table.index[table["z"].abs() > 2]:
        ax_near.annotate(
            f"{number:02d}",
            (number, table.loc[number, "media"]),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
            color=TEXT_SECONDARY,
        )
    ax_near.set_xlim(spec.menor - 0.7, spec.maior + 0.7)
    ax_near.set_xlabel("Dezena x")
    ax_near.set_ylabel("Distância média até a dezena mais próxima")
    ax_near.set_title("Quando sai a dezena x, a que distância fica a mais próxima?")
    ax_near.legend(
        handles=[
            plt.Line2D([], [], color=TEXT_SECONDARY, lw=1.5),
            plt.Rectangle((0, 0), 1, 1, color=REFERENCE, alpha=0.2),
            plt.Rectangle((0, 0), 1, 1, color=SERIES[1]),
            plt.Rectangle((0, 0), 1, 1, color=SERIES[0]),
        ],
        labels=["esperado", "faixa de 95%", "mais longe (z > 2)", "mais perto (z < −2)"],
        loc="upper center",
        ncols=4,
        fontsize=8.5,
    )
    ax_near.set_ylim(0, table["media"].max() * 1.25)
    fig.tight_layout()
    return fig


def plot_neighbor_offsets(df: pd.DataFrame, number: int, spec: GameSpec = MEGA_SENA) -> Figure:
    """Quando sai a dezena ``number``, quantas vezes cada outra dezena saiu junto.

    Sob sorteio uniforme, cada outra dezena acompanha ``number`` com a mesma chance,
    (K - 1)/(N - 1), seja vizinha ou distante: a linha cinza é plana. A distância até
    ``number`` é a distância horizontal até a linha vertical.
    """
    draws = draws_matrix(df)
    times = int((draws == number).any(axis=1).sum())
    offsets = distances.neighbor_offsets(draws, number)
    counts = np.bincount(offsets + number - spec.menor, minlength=spec.n)
    others = spec.numbers[spec.numbers != number]
    p = (spec.sorteadas - 1) / (spec.n - 1)
    expected = times * p
    half_band = 1.96 * np.sqrt(times * p * (1 - p))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhspan(expected - half_band, expected + half_band, color=REFERENCE, alpha=0.15, lw=0)
    near = np.abs(others - number) <= 2
    ax.bar(
        others,
        counts[others - spec.menor],
        color=np.where(near, SERIES[1], SERIES[0]),
        width=0.8,
        edgecolor=SURFACE,
    )
    ax.axhline(expected, color=REFERENCE, lw=1.5, ls="--")
    ax.axvline(number, color=TEXT, lw=2)
    ax.annotate(
        f"{number:02d}",
        (number, 1.0),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="bottom",
        fontweight="bold",
        color=TEXT,
        annotation_clip=False,
    )
    ax.set_xticks(spec.numbers)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(spec.menor - 0.7, spec.maior + 0.7)
    ax.set_xlabel("Dezena que saiu junto")
    ax.set_ylabel("Concursos")
    ax.set_title(
        f"Quando sai a {number:02d} ({times} concursos): quantas vezes cada outra saiu junto",
        pad=18,
    )
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=SERIES[1]),
            plt.Rectangle((0, 0), 1, 1, color=SERIES[0]),
            plt.Line2D([], [], color=REFERENCE, lw=1.5, ls="--"),
        ],
        labels=["até 2 de distância", "demais", f"esperado ({expected:.1f}) e faixa de 95%"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncols=3,
    )
    ax.set_ylim(0, max(counts.max(), expected + half_band) * 1.1)
    return fig


def plot_extra(df: pd.DataFrame, spec: GameSpec) -> Figure:
    """Frequência de cada opção do sorteio extra (Mês da Sorte, Time do Coração).

    A faixa cinza é o intervalo de 95% da Binomial(n_concursos, 1/opções). Meses aparecem em
    ordem de calendário; times, do mais ao menos sorteado. Só entram os concursos com a lista de
    opções fixa (ver :attr:`GameSpec.extra_ate`).
    """
    counts = extra_values(df, spec).value_counts()
    n = int(counts.sum())
    p = 1 / spec.extra_total
    expected = n * p
    half_band = 1.96 * np.sqrt(n * p * (1 - p))
    outside = (counts - expected).abs() > half_band
    title = f"{spec.extra_nome}: frequência de cada opção em {n} concursos"
    if spec.extra_ate is not None:
        title += f" (1 a {spec.extra_ate}, lista fixa de {spec.extra_total})"
    legend = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES[1]),
        plt.Rectangle((0, 0), 1, 1, color=SERIES[0]),
    ]
    labels = ["fora da faixa de 95%", "dentro da faixa"]

    if spec.extra == "mes_da_sorte":
        counts = counts.reindex(MONTHS, fill_value=0)
        outside = outside.reindex(MONTHS, fill_value=True)
        colors = [SERIES[1] if out else SERIES[0] for out in outside]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhspan(expected - half_band, expected + half_band, color=REFERENCE, alpha=0.15, lw=0)
        ax.bar([month[:3] for month in counts.index], counts.values, color=colors, width=0.75)
        ax.axhline(expected, color=REFERENCE, lw=1.5, ls="--")
        for i, value in enumerate(counts.values):
            ax.annotate(
                str(value),
                (i, value),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
                color=TEXT_SECONDARY,
            )
        ax.set_ylim(0, counts.max() * 1.3)  # folga para a legenda
        ax.set_ylabel("Vezes sorteado")
        legend_loc, legend_cols = "upper center", 3
    else:
        counts = counts.sort_values()
        colors = [SERIES[1] if outside[name] else SERIES[0] for name in counts.index]
        fig, ax = plt.subplots(figsize=(10, max(6, 0.2 * len(counts))))
        ax.axvspan(expected - half_band, expected + half_band, color=REFERENCE, alpha=0.15, lw=0)
        ax.barh(counts.index, counts.values, color=colors, height=0.75)
        ax.axvline(expected, color=REFERENCE, lw=1.5, ls="--")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(-0.6, len(counts) - 0.4)
        ax.set_xlabel("Vezes sorteado")
        ax.set_xlim(0, counts.max() * 1.08)
        legend_loc, legend_cols = "lower right", 1  # as barras mais curtas ficam embaixo
    legend.append(plt.Line2D([], [], color=REFERENCE, lw=1.5, ls="--"))
    labels.append(f"esperado: {expected:.0f}")
    ax.legend(handles=legend, labels=labels, loc=legend_loc, ncols=legend_cols)
    ax.set_title(title)
    return fig


def run(spec: GameSpec) -> list[str]:
    """Gera e salva todas as figuras de um jogo.

    A do backtest só é gerada se o backtest já tiver rodado.

    Returns:
        Nomes dos arquivos gerados.
    """
    set_style()
    df = load_processed(spec)
    paths = spec.paths
    paths.figures.mkdir(parents=True, exist_ok=True)

    figures = {
        "frequencia_dezenas.png": plot_number_frequency(df, spec=spec),
        "frequencia_acumulada.png": plot_cumulative_frequency(df, spec=spec),
        "pares_impares.png": plot_even_odd(df, spec),
        "quadrantes.png": plot_quadrants(df, spec),
        "distancias.png": plot_distances(df, spec),
        "vizinhanca.png": plot_neighborhood(df, spec),
    }
    if spec.extra:
        figures[f"{spec.extra}.png"] = plot_extra(df, spec)
    if paths.backtest_summary.exists():
        figures["backtest.png"] = plot_backtest(
            pd.read_csv(paths.backtest_hits),
            pd.read_csv(paths.backtest_null)["media_acertos"].to_numpy(),
            pd.read_csv(paths.backtest_summary),
            spec,
        )
    else:
        logger.warning("Backtest ainda não rodou; pulando backtest.png.")

    for name, fig in figures.items():
        fig.savefig(paths.figures / name)
        plt.close(fig)
        logger.info(f"Figura salva: {paths.figures / name}")
    logger.success(f"{spec.nome}: {len(figures)} figuras geradas.")
    return list(figures)


@app.command()
def main(jogo: str = typer.Option(MEGA_SENA.slug, help="Jogo (slug da API).")):
    """Gera e salva todas as figuras de um jogo."""
    run(get_game(jogo))


if __name__ == "__main__":
    app()
