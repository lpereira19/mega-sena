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
