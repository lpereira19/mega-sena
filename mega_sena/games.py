"""Especificação de cada loteria: regras do sorteio, layout do volante e arquivos.

Todo o código de análise recebe um :class:`GameSpec` em vez de assumir as regras de um jogo
específico. Para incluir uma loteria nova, basta registrar a especificação dela em :data:`GAMES`.

As análises valem para jogos em que, a cada concurso, ``sorteadas`` dezenas distintas são
sorteadas sem reposição entre ``menor`` e ``maior``, e a aposta mínima marca ``aposta`` dezenas
(``aposta >= sorteadas``).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mega_sena.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

API_BASE_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api"


@dataclass(frozen=True)
class GamePaths:
    """Arquivos de entrada e saída de um jogo."""

    raw: Path
    processed: Path
    features: Path
    hypothesis_tests: Path
    distance_tests: Path
    backtest_summary: Path
    backtest_hits: Path
    backtest_null: Path
    figures: Path


@dataclass(frozen=True)
class GameSpec:
    """Regras de uma loteria de dezenas sorteadas sem reposição.

    Attributes:
        slug: Nome do jogo na API da Caixa e prefixo dos arquivos (ex.: ``"megasena"``).
        nome: Nome para exibição (ex.: ``"Mega-Sena"``).
        menor: Menor dezena do volante (0 na Lotomania, 1 nos demais).
        maior: Maior dezena do volante.
        sorteadas: Dezenas sorteadas por concurso (K).
        aposta: Dezenas marcadas na aposta mínima (B), usada no backtest.
        volante: Layout do volante impresso, ``(linhas, colunas)``, usado nos quadrantes.
        faixas: Faixas de prêmio contadas no backtest, como ``(coluna, acertos)``.
        periodos: Janelas de anos da análise exploratória por período (``None`` no fim = até o
            ano mais recente).
        teto_salto: Saltos entre dezenas vizinhas a partir deste valor ficam numa categoria só.
    """

    slug: str
    nome: str
    menor: int
    maior: int
    sorteadas: int
    aposta: int
    volante: tuple[int, int]
    faixas: tuple[tuple[str, int], ...] = ()
    periodos: tuple[tuple[int, int | None], ...] = ()
    teto_salto: int = 25

    def __post_init__(self):
        rows, cols = self.volante
        if rows * cols != self.n:
            raise ValueError(f"{self.nome}: volante {rows}x{cols} não tem {self.n} dezenas.")
        if self.aposta < self.sorteadas:
            raise ValueError(f"{self.nome}: a aposta precisa ter pelo menos {self.sorteadas}.")
        max_gap = self.n - self.sorteadas + 1
        if not 2 <= self.teto_salto <= max_gap:
            raise ValueError(f"{self.nome}: teto_salto deve estar entre 2 e {max_gap}.")

    @property
    def n(self) -> int:
        """Total de dezenas do volante (N)."""
        return self.maior - self.menor + 1

    @property
    def numbers(self) -> np.ndarray:
        """Todas as dezenas do volante, em ordem."""
        return np.arange(self.menor, self.maior + 1)

    @property
    def draw_columns(self) -> list[str]:
        """Colunas das dezenas em ordem crescente no CSV processado (``d1``, ``d2``...)."""
        return [f"d{i}" for i in range(1, self.sorteadas + 1)]

    @property
    def order_columns(self) -> list[str]:
        """Colunas das dezenas na ordem de sorteio no CSV processado (``bola1``...)."""
        return [f"bola{i}" for i in range(1, self.sorteadas + 1)]

    @property
    def api_url(self) -> str:
        return f"{API_BASE_URL}/{self.slug}"

    @property
    def paths(self) -> GamePaths:
        return GamePaths(
            raw=RAW_DATA_DIR / f"{self.slug}_api.jsonl",
            processed=PROCESSED_DATA_DIR / f"{self.slug}.csv",
            features=PROCESSED_DATA_DIR / f"{self.slug}_features.csv",
            hypothesis_tests=REPORTS_DIR / self.slug / "hypothesis_tests.csv",
            distance_tests=REPORTS_DIR / self.slug / "distance_tests.csv",
            backtest_summary=MODELS_DIR / self.slug / "backtest_summary.csv",
            backtest_hits=MODELS_DIR / self.slug / "backtest_hits.csv",
            backtest_null=MODELS_DIR / self.slug / "backtest_null_means.csv",
            figures=FIGURES_DIR / self.slug,
        )


MEGA_SENA = GameSpec(
    slug="megasena",
    nome="Mega-Sena",
    menor=1,
    maior=60,
    sorteadas=6,
    aposta=6,
    volante=(6, 10),
    faixas=(("quadras", 4), ("quinas", 5), ("senas", 6)),
    periodos=((1996, 2005), (2006, 2015), (2016, None)),
    teto_salto=25,
)

GAMES: dict[str, GameSpec] = {spec.slug: spec for spec in [MEGA_SENA]}


def get_game(slug: str) -> GameSpec:
    """Retorna a especificação do jogo pelo slug.

    Raises:
        KeyError: Se o jogo não estiver registrado.
    """
    try:
        return GAMES[slug]
    except KeyError:
        raise KeyError(f"Jogo {slug!r} desconhecido. Disponíveis: {', '.join(GAMES)}.") from None
