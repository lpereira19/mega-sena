"""Jogos registrados e sorteios extras (Mês da Sorte e Time do Coração)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mega_sena import distances
from mega_sena.dataset import (
    MONTHS,
    build_dataframe,
    normalize_extra,
    parse_contest,
    resolve_teams,
)
from mega_sena.features import quadrant_of, quadrant_sizes
from mega_sena.games import (
    DIA_DE_SORTE,
    DUPLA_SENA,
    GAMES,
    MAIS_MILIONARIA,
    TIMEMANIA,
    GameSpec,
)
from mega_sena.hypothesis import (
    extra_as_game,
    extra_combination_tests,
    extra_test,
    overlap_test,
    random_draws,
)
from mega_sena.modeling.backtest import hits_pmf
from mega_sena.plots import plot_extra, plot_quadrants


@pytest.mark.parametrize("spec", GAMES.values(), ids=lambda s: s.slug)
def test_registered_games_have_valid_distributions(spec):
    assert quadrant_sizes(spec).sum() == spec.n
    assert hits_pmf(spec).sum() == pytest.approx(1)
    for pmf in [
        distances.consecutive_pairs_pmf(spec),
        distances.longest_run_pmf(spec),
        distances.min_gap_pmf(spec),
        distances.gap_pmf(spec),
        distances.pair_distance_pmf(spec),
    ]:
        assert pmf.sum() == pytest.approx(1)
    assert all(column for column, _ in spec.faixas)
    assert all(0 <= k <= spec.sorteadas for _, k in spec.faixas)


def test_partial_volante_is_accepted_but_not_oversized():
    # Dia de Sorte: 31 dezenas numa grade 5x7 com a última linha incompleta (29, 30, 31).
    assert quadrant_sizes(DIA_DE_SORTE).tolist() == [6, 8, 9, 8]
    assert quadrant_of(31, DIA_DE_SORTE) == 3
    with pytest.raises(ValueError):
        GameSpec("x", "linha sobrando", 1, 31, 7, 7, (6, 7))


@pytest.mark.parametrize(
    ("raw", "month"),
    [("2", "Fevereiro"), ("12", "Dezembro"), ("Março", "Março"), ("MARCO", "Março"), (" abril ", "Abril")],
)
def test_month_is_normalized(raw, month):
    assert normalize_extra(raw, DIA_DE_SORTE) == month


def test_unknown_month_raises():
    with pytest.raises(ValueError):
        normalize_extra("13", DIA_DE_SORTE)
    with pytest.raises(ValueError):
        normalize_extra("Brumário", DIA_DE_SORTE)


def test_team_spaces_are_collapsed_and_empty_is_none():
    assert normalize_extra("  FLAMENGO   /RJ ", TIMEMANIA) == "FLAMENGO /RJ"
    assert normalize_extra("", TIMEMANIA) is None
    assert normalize_extra(None, DIA_DE_SORTE) is None


def test_build_dataframe_keeps_the_extra_column():
    records = [
        {
            "numero": i,
            "dataApuracao": "19/05/2018",
            "listaDezenas": ["01", "05", "09", "13", "17", "21", "31"],
            "nomeTimeCoracaoMesSorte": month,
        }
        for i, month in enumerate(["2", "Fevereiro"], start=1)
    ]
    df = build_dataframe(records, DIA_DE_SORTE)
    assert df["mes_da_sorte"].tolist() == ["Fevereiro", "Fevereiro"]
    assert df.loc[0, "d7"] == 31


def _fake_months(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"mes_da_sorte": rng.choice(MONTHS, size=n)})


def test_extra_test_is_calibrated():
    p_values = [extra_test(_fake_months(500, seed), DIA_DE_SORTE).p_valor for seed in range(400)]
    assert 0.02 < np.mean(np.array(p_values) < 0.05) < 0.09


def test_extra_test_detects_bias_and_counts_missing_options():
    biased = pd.DataFrame({"mes_da_sorte": ["Janeiro"] * 300 + list(MONTHS[1:]) * 10})
    assert extra_test(biased, DIA_DE_SORTE).p_valor < 1e-6
    # Só 11 meses aparecem: o 12º entra com contagem zero, e o teste continua com 12 categorias.
    missing = pd.DataFrame({"mes_da_sorte": list(MONTHS[:11]) * 50})
    assert extra_test(missing, DIA_DE_SORTE).p_valor < 1e-6


def test_extra_test_rejects_more_options_than_the_game_has():
    too_many = pd.DataFrame({"mes_da_sorte": [f"mes {i}" for i in range(13)]})
    with pytest.raises(ValueError):
        extra_test(too_many, DIA_DE_SORTE)


def test_dia_de_sorte_figures_render():
    rng = np.random.default_rng(1)
    draws = np.sort(random_draws(200, rng, DIA_DE_SORTE), axis=1)
    df = pd.DataFrame(draws, columns=DIA_DE_SORTE.draw_columns)
    df.insert(0, "concurso", np.arange(1, len(df) + 1))
    df.insert(1, "data", pd.date_range("2018-05-19", periods=len(df), freq="3D"))
    df["mes_da_sorte"] = rng.choice(MONTHS, size=len(df))
    for fig in [plot_quadrants(df, DIA_DE_SORTE), plot_extra(df, DIA_DE_SORTE)]:
        plt.close(fig)


NUL = chr(0)


def test_resolve_teams_unifies_spellings_and_repairs_corrupted_records():
    raw = pd.Series(
        [
            "AVAÍ/SC",
            "AVAI /SC",
            "AMERICA /RJ",
            "AMERICANO/RJ",
            f"AMERICANOEABA{NUL * 4}/RJ",  # lixo depois do nome: o prefixo conhecido mais longo
            f"AMERICAUJOAOA{NUL * 4}/RJ",
            "SAMP CORREA /MA",
            "SAMPAIO CORRÊA/MA",
            "JI-PARANÁ / RO",
            f"XYZ{NUL * 4}/RO",  # nenhum time conhecido como prefixo
            None,
        ]
    )
    assert resolve_teams(raw).fillna("-").tolist() == [
        "AVAI/SC",
        "AVAI/SC",
        "AMERICA/RJ",
        "AMERICANO/RJ",
        "AMERICANO/RJ",
        "AMERICA/RJ",
        "SAMPAIO CORREA/MA",
        "SAMPAIO CORREA/MA",
        "JI-PARANA/RO",
        "-",
        "-",
    ]


def test_extra_test_only_uses_the_fixed_list_window():
    spec = GameSpec("x", "Teste", 1, 80, 7, 10, (8, 10), extra="time", extra_total=2, extra_ate=40)
    # Até o concurso 40, A e B empatados; depois, só times novos (que o teste deve ignorar).
    times = ["A", "B"] * 20 + ["C", "D", "E", "F"] * 10
    df = pd.DataFrame({"concurso": range(1, len(times) + 1), "time": times})
    result = extra_test(df, spec)
    assert result.p_valor == pytest.approx(1.0)
    assert "concursos até 40" in result.observacao


def test_parse_mais_milionaria_separates_numbers_and_trevos():
    record = {
        "numero": 1,
        "dataApuracao": "28/05/2022",
        "listaDezenas": ["01", "03", "07", "15", "23", "44"],
        "trevosSorteados": ["4", "2"],
        "dezenasSorteadasOrdemSorteio": ["23", "44", "01", "07", "03", "15", "04", "02"],
    }
    row = parse_contest(record, MAIS_MILIONARIA)
    assert row["trevos"] == "2-4"
    assert [row[f"bola{i}"] for i in range(1, 7)] == [23, 44, 1, 7, 3, 15]


def _fake_trevos(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = np.sort(random_draws(n, rng, extra_as_game(MAIS_MILIONARIA)), axis=1)
    return pd.DataFrame({"concurso": range(1, n + 1), "trevos": [f"{a}-{b}" for a, b in pairs]})


def test_trevo_tests_are_calibrated():
    p_values = np.array(
        [[r.p_valor for r in extra_combination_tests(_fake_trevos(300, s), MAIS_MILIONARIA)]
         for s in range(300)]
    )
    rejection = (p_values < 0.05).mean(axis=0)
    assert ((0.02 < rejection) & (rejection < 0.09)).all()


def test_trevo_combination_test_sees_dependence_the_marginals_miss():
    # Só os pares 1-2, 3-4 e 5-6: cada trevo sai exatamente 1/3 das vezes (marginais perfeitas),
    # mas as combinações estão longe de uniformes.
    df = pd.DataFrame({"concurso": range(1, 301), "trevos": ["1-2", "3-4", "5-6"] * 100})
    by_option, by_combo = extra_combination_tests(df, MAIS_MILIONARIA)
    assert by_option.p_valor > 0.99
    assert by_combo.p_valor < 1e-10


def _fake_dupla(n: int, seed: int, copy_first: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    first = np.sort(random_draws(n, rng, DUPLA_SENA), axis=1)
    second = first if copy_first else np.sort(random_draws(n, rng, DUPLA_SENA), axis=1)
    rows = [
        {"concurso": c, "sorteio": s, **dict(zip(DUPLA_SENA.draw_columns, draws[c - 1]))}
        for c in range(1, n + 1)
        for s, draws in [(1, first), (2, second)]
    ]
    return pd.DataFrame(rows)


def test_overlap_test_is_calibrated_and_detects_dependence():
    p_values = [overlap_test(_fake_dupla(400, s), DUPLA_SENA).p_valor for s in range(300)]
    assert 0.02 < np.mean(np.array(p_values) < 0.05) < 0.09
    assert overlap_test(_fake_dupla(100, 0, copy_first=True), DUPLA_SENA).p_valor < 1e-10


def _dupla_record(numero: int, tiers: list[tuple[int, str, int]]) -> dict:
    return {
        "numero": numero,
        "dataApuracao": "09/04/2016",
        "listaDezenas": ["12", "14", "22", "33", "36", "50"],
        "listaDezenasSegundoSorteio": ["18", "20", "29", "32", "41", "49"],
        "dezenasSorteadasOrdemSorteio": [
            "22", "36", "33", "12", "14", "50", "20", "49", "18", "32", "29", "41",
        ],
        "listaRateioPremio": [
            {"faixa": f, "descricaoFaixa": d, "numeroDeGanhadores": g} for f, d, g in tiers
        ],
    }


def test_dupla_sena_has_one_row_per_draw_and_finds_the_second_jackpot():
    # Sem a faixa de 3 acertos, a sena do 2º sorteio é a faixa 5; com ela, também, mas a busca
    # é pela descrição, não pelo número.
    tiers = [(1, "6 acertos", 0), (2, "5 acertos", 26), (3, "4 acertos", 1930),
             (5, "6 acertos", 2), (6, "5 acertos", 27), (7, "4 acertos", 1449)]
    df = build_dataframe([_dupla_record(1, tiers), _dupla_record(2, tiers)], DUPLA_SENA)
    assert df[["concurso", "sorteio"]].values.tolist() == [[1, 1], [1, 2], [2, 1], [2, 2]]
    first, second = df.iloc[0], df.iloc[1]
    assert [first[f"d{i}"] for i in range(1, 7)] == [12, 14, 22, 33, 36, 50]
    assert [second[f"bola{i}"] for i in range(1, 7)] == [20, 49, 18, 32, 29, 41]
    assert (first["ganhadores_premio_maximo"], second["ganhadores_premio_maximo"]) == (0, 2)
