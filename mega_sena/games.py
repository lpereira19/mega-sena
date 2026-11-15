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
        volante: Layout do volante, ``(linhas, colunas)``, usado nos quadrantes. As dezenas
            preenchem a grade linha a linha; a última linha pode ficar incompleta (Dia de Sorte,
            31 dezenas numa grade de calendário com 7 colunas).
        faixas: Faixas de prêmio contadas no backtest, como ``(coluna, acertos)``.
        periodos: Janelas de anos da análise exploratória por período (``None`` no fim = até o
            ano mais recente).
        teto_salto: Saltos entre dezenas vizinhas a partir deste valor ficam numa categoria só.
        extra: Coluna do sorteio extra no CSV processado (``"mes_da_sorte"`` no Dia de Sorte,
            ``"time_do_coracao"`` na Timemania), ou ``None`` se o jogo não tem.
        extra_nome: Nome do sorteio extra para exibição.
        extra_total: Quantas opções o sorteio extra tem (12 meses, 80 times).
        extra_ate: Último concurso em que a lista de opções do sorteio extra foi fixa, ou
            ``None`` se nunca mudou. O teste e a figura do sorteio extra usam só os concursos
            até aqui: depois, a Timemania passou a trocar clubes da lista, e sem saber quais
            valiam em cada concurso não dá para calcular a frequência esperada.
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
    extra: str | None = None
    extra_nome: str = ""
    extra_total: int = 0
    extra_ate: int | None = None

    def __post_init__(self):
        rows, cols = self.volante
        if not (rows - 1) * cols < self.n <= rows * cols:
            raise ValueError(f"{self.nome}: volante {rows}x{cols} não comporta {self.n} dezenas.")
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

LOTOFACIL = GameSpec(
    slug="lotofacil",
    nome="Lotofácil",
    menor=1,
    maior=25,
    sorteadas=15,
    aposta=15,
    volante=(5, 5),
    faixas=tuple((f"{k}_pontos", k) for k in range(11, 16)),
    periodos=((2003, 2012), (2013, 2019), (2020, None)),
    teto_salto=6,
)

QUINA = GameSpec(
    slug="quina",
    nome="Quina",
    menor=1,
    maior=80,
    sorteadas=5,
    aposta=5,
    volante=(8, 10),
    faixas=(("duques", 2), ("ternos", 3), ("quadras", 4), ("quinas", 5)),
    periodos=((1994, 2005), (2006, 2015), (2016, None)),
    teto_salto=40,
)

LOTOMANIA = GameSpec(
    slug="lotomania",
    nome="Lotomania",
    menor=0,
    maior=99,
    sorteadas=20,
    aposta=50,
    volante=(10, 10),
    faixas=(("0_pontos", 0), *((f"{k}_pontos", k) for k in range(15, 21))),
    periodos=((1999, 2009), (2010, 2017), (2018, None)),
    teto_salto=20,
)

TIMEMANIA = GameSpec(
    slug="timemania",
    nome="Timemania",
    menor=1,
    maior=80,
    sorteadas=7,
    aposta=10,
    volante=(8, 10),
    faixas=tuple((f"{k}_pontos", k) for k in range(3, 8)),
    periodos=((2008, 2013), (2014, 2019), (2020, None)),
    teto_salto=35,
    extra="time_do_coracao",
    extra_nome="Time do Coração",
    extra_total=80,
    extra_ate=1781,  # a partir do concurso 1782 (12/05/2022) a lista de clubes mudou
)

DIA_DE_SORTE = GameSpec(
    slug="diadesorte",
    nome="Dia de Sorte",
    menor=1,
    maior=31,
    sorteadas=7,
    aposta=7,
    volante=(5, 7),
    faixas=tuple((f"{k}_pontos", k) for k in range(4, 8)),
    periodos=((2018, 2020), (2021, 2023), (2024, None)),
    teto_salto=15,
    extra="mes_da_sorte",
    extra_nome="Mês da Sorte",
    extra_total=12,
)

GAMES: dict[str, GameSpec] = {
    spec.slug: spec for spec in [MEGA_SENA, LOTOFACIL, QUINA, LOTOMANIA, TIMEMANIA, DIA_DE_SORTE]
}


def get_game(slug: str) -> GameSpec:
    """Retorna a especificação do jogo pelo slug.

    Raises:
        KeyError: Se o jogo não estiver registrado.
    """
    try:
        return GAMES[slug]
    except KeyError:
        raise KeyError(f"Jogo {slug!r} desconhecido. Disponíveis: {', '.join(GAMES)}.") from None
