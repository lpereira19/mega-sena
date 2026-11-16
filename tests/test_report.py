import pandas as pd
import pytest

from mega_sena.report import (
    END_MARKER,
    START_MARKER,
    backtest_reading,
    fmt_int,
    fmt_p,
    update_readme,
)


def test_brazilian_formatting():
    assert fmt_int(3080) == "3.080"
    assert fmt_p(0.0069) == "0,007"
    assert fmt_p(0.0004) == "< 0,001"


def test_update_readme_only_touches_marked_section(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"antes\n{START_MARKER}\nvelho\n{END_MARKER}\ndepois\n", encoding="utf-8")
    update_readme("novo", readme)
    text = readme.read_text(encoding="utf-8")
    assert "novo" in text and "velho" not in text
    assert text.startswith("antes\n") and text.endswith("depois\n")


def test_update_readme_requires_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("sem marcadores", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme("novo", readme)


def _summary(p_mc: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estrategia": [f"e{i}" for i in range(len(p_mc))],
            "apostas": 100,
            "media_acertos": 0.6,
            "quinas": 0,
            "senas": 0,
            "p_valor_monte_carlo": p_mc,
            "p_valor_mc_bonferroni": [min(1.0, p * len(p_mc)) for p in p_mc],
        }
    )


def test_backtest_reading_wording_follows_results():
    assert "Nenhuma estratégia se distingue" in backtest_reading(_summary([0.5, 0.03]))
    assert '"e1" parece promissora' in backtest_reading(_summary([0.5, 0.03]))
    assert "mesmo após a correção" in backtest_reading(_summary([0.5, 0.001]))


def test_game_rule_describes_each_game():
    from mega_sena.games import DUPLA_SENA, LOTOMANIA, MAIS_MILIONARIA, MEGA_SENA, TIMEMANIA
    from mega_sena.report import game_rule

    assert game_rule(MEGA_SENA) == "6 de 60"
    assert game_rule(LOTOMANIA) == "20 de 100, aposta 50"
    assert game_rule(TIMEMANIA) == "7 de 80, aposta 10 + Time do Coração"
    assert game_rule(DUPLA_SENA) == "6 de 50, 2 sorteios"
    assert game_rule(MAIS_MILIONARIA) == "6 de 50 + 2 de 6 trevos"


def _fake_results(spec, p_freq: float, p_hot: float):
    from mega_sena.hypothesis import (
        EVEN_ODD_TEST,
        FREQUENCY_MC_TEST,
        FREQUENCY_TEST,
        QUADRANT_TEST,
        format_test_name,
    )
    from mega_sena.report import GameResults

    tests = pd.DataFrame(
        {
            "teste": [
                format_test_name(FREQUENCY_TEST, spec),
                format_test_name(FREQUENCY_MC_TEST, spec),
                EVEN_ODD_TEST,
                QUADRANT_TEST,
                format_test_name(FREQUENCY_TEST, spec) + ", 2020-2026",
            ],
            "p_valor": [p_freq, p_freq, 0.5, 0.5, 1e-9],
            "observacao": ["", "", "", "", "pós-hoc; 100 concursos"],
        }
    )
    distance_tests = pd.DataFrame(
        {"teste": ["Pares consecutivos por concurso"], "p_valor": [0.5], "p_valor_bonferroni": [1.0]}
    )
    summary = pd.DataFrame(
        {
            "estrategia": ["quentes (histórico)", "aleatória"],
            "p_valor_monte_carlo": [p_hot, 0.5],
            "p_valor_mc_bonferroni": [min(1.0, 2 * p_hot), 1.0],
        }
    )
    df = pd.DataFrame({"concurso": [1, 2], "data": pd.to_datetime(["2001-01-01", "2002-01-01"])})
    return GameResults(spec, df, tests, distance_tests, summary)


def test_comparison_counts_only_confirmatory_tests_and_applies_global_bonferroni():
    from mega_sena.games import LOTOFACIL, MEGA_SENA
    from mega_sena.report import confirmatory_p_values, render_comparison

    results = [_fake_results(MEGA_SENA, 0.01, 0.03), _fake_results(LOTOFACIL, 1e-6, 0.5)]
    p_values = confirmatory_p_values(results)
    # Por jogo: frequência, pares, quadrantes, 1 de distância e 2 estratégias. O Monte Carlo
    # da frequência e o recorte por período (p = 1e-9) ficam de fora.
    assert len(p_values) == 12
    assert p_values["p_valor"].min() == 1e-6

    text = render_comparison(results)
    assert "| Mega-Sena | 6 de 60 | 2 (desde 2001) | **0,010** |" in text
    assert "sobrevive só o teste de frequência das 25 dezenas da Lotofácil" in text
    assert "um deles é uma estratégia do backtest" in text  # a "quentes" da Mega-Sena
    assert "Mega-Sena (p = 0,010), destaque da seção anterior, não passa" in text
