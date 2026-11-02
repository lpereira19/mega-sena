import pandas as pd
import pytest

from mega_sena.dataset import build_dataframe, load_results_file, parse_contest, to_long


def make_record(numero: int, dezenas: list[str], ordem: list[str] | None = None) -> dict:
    """Registro mínimo no formato da API da Caixa."""
    return {
        "numero": numero,
        "dataApuracao": "11/03/1996",
        "listaDezenas": dezenas,
        "dezenasSorteadasOrdemSorteio": ordem,
        "listaRateioPremio": [{"faixa": 1, "numeroDeGanhadores": 2, "valorPremio": 1.0}],
        "acumulado": False,
        "valorArrecadado": 1000.0,
    }


def test_parse_contest_types_and_order():
    row = parse_contest(make_record(1, ["04", "05", "30", "33", "41", "52"],
                                    ["41", "05", "04", "52", "30", "33"]))
    assert row["concurso"] == 1
    assert row["data"] == pd.Timestamp("1996-03-11")
    assert [row[f"d{i}"] for i in range(1, 7)] == [4, 5, 30, 33, 41, 52]
    assert [row[f"bola{i}"] for i in range(1, 7)] == [41, 5, 4, 52, 30, 33]
    assert row["ganhadores_sena"] == 2


def test_parse_contest_without_draw_order_falls_back_to_sorted():
    row = parse_contest(make_record(1, ["01", "02", "03", "04", "05", "06"]))
    assert [row[f"bola{i}"] for i in range(1, 7)] == [1, 2, 3, 4, 5, 6]


def test_build_dataframe_sorts_and_deduplicates():
    records = [
        make_record(2, ["01", "02", "03", "04", "05", "06"]),
        make_record(1, ["07", "08", "09", "10", "11", "12"]),
        make_record(2, ["01", "02", "03", "04", "05", "06"]),
    ]
    df = build_dataframe(records)
    assert df["concurso"].tolist() == [1, 2]


@pytest.mark.parametrize(
    "dezenas",
    [
        ["00", "02", "03", "04", "05", "06"],  # fora do intervalo
        ["61", "02", "03", "04", "05", "06"],  # fora do intervalo
        ["01", "01", "03", "04", "05", "06"],  # repetida
    ],
)
def test_build_dataframe_rejects_invalid_draws(dezenas):
    with pytest.raises(ValueError):
        build_dataframe([make_record(1, dezenas)])


def test_to_long_has_one_row_per_number():
    df = build_dataframe([make_record(1, ["04", "05", "30", "33", "41", "52"],
                                      ["41", "05", "04", "52", "30", "33"])])
    long = to_long(df)
    assert len(long) == 6
    assert long["dezena"].tolist() == [41, 5, 4, 52, 30, 33]
    assert long["posicao"].tolist() == [1, 2, 3, 4, 5, 6]


def test_load_results_file_caixa_layout(tmp_path):
    path = tmp_path / "mega.csv"
    path.write_text(
        "Concurso,Data do Sorteio,Bola1,Bola2,Bola3,Bola4,Bola5,Bola6\n"
        "1,11/03/1996,41,5,4,52,30,33\n",
        encoding="utf-8",
    )
    df = load_results_file(path)
    assert df.loc[0, "data"] == pd.Timestamp("1996-03-11")
    assert df.loc[0, ["d1", "d2", "d3", "d4", "d5", "d6"]].tolist() == [4, 5, 30, 33, 41, 52]
