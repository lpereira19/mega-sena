"""Textos de resultados gerados a partir dos números, para o README e o notebook.

Os números da análise mudam a cada concurso novo. Para os textos que os citam nunca ficarem
desatualizados, eles são montados aqui, inclusive as palavras que dependem do resultado (por
exemplo, "rejeita" ou "não rejeita"). No README, só o trecho entre :data:`START_MARKER` e
:data:`END_MARKER` é reescrito; o restante continua editável à mão.

Uso::

    python -m mega_sena.report   # reescreve a seção de resultados do README
"""

from pathlib import Path
import re

from loguru import logger
import pandas as pd
import typer

from mega_sena.config import (
    BACKTEST_SUMMARY_FILE,
    HYPOTHESIS_TESTS_FILE,
    NUMBERS_PER_DRAW,
    PROCESSED_RESULTS_FILE,
    PROJ_ROOT,
    TOTAL_NUMBERS,
)
from mega_sena.dataset import load_processed
from mega_sena.features import draws_matrix, number_frequencies
from mega_sena.hypothesis import (
    EVEN_ODD_TEST,
    FREQUENCY_MC_TEST,
    FREQUENCY_TEST,
    QUADRANT_TEST,
)
from mega_sena.modeling.backtest import EXPECTED_HITS

app = typer.Typer()

README_FILE = PROJ_ROOT / "README.md"
START_MARKER = "<!-- RESULTADOS:INICIO (gerado por mega_sena.report; não edite à mão) -->"
END_MARKER = "<!-- RESULTADOS:FIM -->"
ALPHA = 0.05
HOT_STRATEGY = "quentes (histórico)"


# ---------------------------------------------------------------------------------------------
# Formatação (padrão brasileiro: vírgula decimal, ponto de milhar)
# ---------------------------------------------------------------------------------------------


def fmt_int(value: int) -> str:
    """``3080`` -> ``"3.080"``."""
    return f"{value:,}".replace(",", ".")


def fmt_num(value: float, digits: int = 3) -> str:
    """``0.6053`` -> ``"0,605"``."""
    return f"{value:.{digits}f}".replace(".", ",")


def fmt_p(p: float) -> str:
    """P-valor com 3 casas; valores muito pequenos viram ``"< 0,001"``."""
    return "< 0,001" if p < 0.001 else fmt_num(p)


def p_eq(p: float) -> str:
    """P-valor para uso em frase: ``"p = 0,007"`` ou ``"p < 0,001"``."""
    formatted = fmt_p(p)
    return f"p {formatted}" if formatted.startswith("<") else f"p = {formatted}"


def _p_value(tests: pd.DataFrame, name: str) -> float:
    return float(tests.loc[tests["teste"] == name, "p_valor"].iloc[0])


def _join(items: list[str]) -> str:
    """``["a", "b", "c"]`` -> ``"a, b e c"``."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} e {items[-1]}"


# ---------------------------------------------------------------------------------------------
# Blocos de texto
# ---------------------------------------------------------------------------------------------


def period_table(tests: pd.DataFrame) -> pd.DataFrame:
    """Linhas exploratórias do teste de frequência por período."""
    rows = tests[tests["teste"].str.startswith(f"{FREQUENCY_TEST}, ")]
    return pd.DataFrame(
        {
            "periodo": rows["teste"].str.removeprefix(f"{FREQUENCY_TEST}, ").str.replace("-", "–"),
            "concursos": rows["observacao"].str.extract(r"(\d+) concursos")[0].astype(int),
            "p_valor": rows["p_valor"],
        }
    ).reset_index(drop=True)


def frequency_reading(df: pd.DataFrame, tests: pd.DataFrame) -> str:
    """Parágrafo sobre a uniformidade das frequências das 60 dezenas."""
    freq = number_frequencies(draws_matrix(df))
    expected = len(df) * NUMBERS_PER_DRAW / TOTAL_NUMBERS
    p_freq, p_mc = _p_value(tests, FREQUENCY_TEST), _p_value(tests, FREQUENCY_MC_TEST)
    top, bottom = int(freq.idxmax()), int(freq.idxmin())

    if p_freq < ALPHA:
        mc = "confirma" if p_mc < ALPHA else "não confirma"
        text = (
            "**A frequência das dezenas não é perfeitamente uniforme.** O teste global rejeita "
            f"H0 ({p_eq(p_freq)}), e o teste de Monte Carlo, que não depende de "
            f"aproximações, {mc} ({p_eq(p_mc)})."
        )
    else:
        text = (
            "**A frequência das dezenas é compatível com um sorteio uniforme** "
            f"({p_eq(p_freq)}; Monte Carlo: {p_eq(p_mc)})."
        )
    return (
        f"{text} A dezena {top} saiu {freq[top]} vezes e a {bottom} saiu {freq[bottom]}, "
        f"contra {expected:.0f} esperadas."
    )


def period_reading(tests: pd.DataFrame) -> str:
    """Tabela por período e o aviso de que a análise é exploratória."""
    periods = period_table(tests)
    lines = ["| Período | Concursos | p-valor |", "|---|---|---|"]
    lines += [
        f"| {row.periodo} | {fmt_int(row.concursos)} | {fmt_p(row.p_valor)} |"
        for row in periods.itertuples()
    ]
    lowest = periods.loc[periods["p_valor"].idxmin()]
    caveat = (
        "Essa quebra é **exploratória**: as janelas foram escolhidas depois de ver o teste "
        "global, então os p-valores por período não são confirmatórios."
    )
    if lowest["p_valor"] < ALPHA:
        caveat += (
            f" O desvio se concentra em {lowest['periodo']} ({p_eq(lowest['p_valor'])}). "
            "Isso sugere hipóteses, como mudança de equipamento ou viés físico das bolas, mas "
            "não as confirma: com os dados disponíveis, não dá para separar um viés real de um "
            "evento raro do acaso."
        )
    return "\n".join(lines) + "\n\n" + caveat


def structure_reading(tests: pd.DataFrame) -> str:
    """Frase sobre pares/ímpares e quadrantes."""
    p_even, p_quad = _p_value(tests, EVEN_ODD_TEST), _p_value(tests, QUADRANT_TEST)
    even = "segue" if p_even >= ALPHA else "**não** segue"
    quad = "recebem" if p_quad >= ALPHA else "**não** recebem"
    return (
        f"A composição de pares e ímpares {even} a hipergeométrica prevista pela teoria "
        f"({p_eq(p_even)}), e os quadrantes {quad} 25% das dezenas cada "
        f"({p_eq(p_quad)})."
    )


def hypothesis_reading(df: pd.DataFrame, tests: pd.DataFrame) -> str:
    """Leitura completa dos testes de hipótese (usada no notebook)."""
    return "\n\n".join(
        [frequency_reading(df, tests), structure_reading(tests), period_reading(tests)]
    )


def backtest_reading(summary: pd.DataFrame) -> str:
    """Tabela e leitura do backtest, com a correção de Bonferroni."""
    lines = [
        "| Estratégia | Acertos por concurso | p (Monte Carlo) | p (Bonferroni) |",
        "|---|---|---|---|",
        f"| *esperado ao acaso* | *{fmt_num(EXPECTED_HITS)}* | | |",
    ]
    lines += [
        f"| {row.estrategia} | {fmt_num(row.media_acertos)} | "
        f"{fmt_p(row.p_valor_monte_carlo)} | {fmt_p(row.p_valor_mc_bonferroni)} |"
        for row in summary.itertuples()
    ]

    significant = summary[summary["p_valor_mc_bonferroni"] < ALPHA]
    nominal = summary[
        (summary["p_valor_monte_carlo"] < ALPHA) & (summary["p_valor_mc_bonferroni"] >= ALPHA)
    ]
    if significant.empty:
        text = "**Nenhuma estratégia se distingue do acaso** depois da correção para múltiplas comparações."
        if not nominal.empty:
            names = _join([f'"{name}"' for name in nominal["estrategia"]])
            text += (
                f" {names} parece{'m' if len(nominal) > 1 else ''} promissora"
                f"{'s' if len(nominal) > 1 else ''} isoladamente (p < 0,05), mas foram testadas "
                f"{len(summary)} estratégias: alguma sempre pode parecer boa por sorte, e com "
                "Bonferroni a vantagem some."
            )
    else:
        names = _join([f'"{name}"' for name in significant["estrategia"]])
        text = (
            f"**{names} se distingue{'m' if len(significant) > 1 else ''} do acaso mesmo após "
            "a correção de Bonferroni.** Antes de concluir, vale checar o tamanho do efeito: "
            "a diferença em relação a 0,6 acerto por concurso é pequena demais para ter valor "
            "prático?"
        )

    jackpots = int(summary["quinas"].sum() + summary["senas"].sum())
    bets = int(summary["apostas"].iloc[0])
    if jackpots == 0:
        text += f" Nenhuma estratégia acertou sequer uma quina em {fmt_int(bets)} apostas."
    else:
        text += f" Somadas, as estratégias fizeram {jackpots} quina(s) ou sena(s) em {fmt_int(bets)} apostas cada."
    return "\n".join(lines) + "\n\n" + text


def conclusions(df: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Lista de conclusões (usada no notebook)."""
    p_freq = _p_value(tests, FREQUENCY_TEST)
    any_strategy = (summary["p_valor_mc_bonferroni"] < ALPHA).any()
    freq = (
        "A frequência das dezenas **não** é perfeitamente uniforme no histórico completo."
        if p_freq < ALPHA
        else "A frequência das dezenas é compatível com um sorteio uniforme."
    )
    strategy = (
        "Alguma estratégia se destacou mesmo após a correção para múltiplas comparações: "
        "investigue o tamanho do efeito antes de qualquer conclusão."
        if any_strategy
        else "Nenhuma estratégia baseada no histórico vence o acaso depois da correção para "
        "múltiplas comparações."
    )
    lesson = (
        "Significância estatística não é o mesmo que efeito explorável."
        if p_freq < ALPHA and not any_strategy
        else "O tamanho do efeito importa tanto quanto o p-valor."
    )
    return "\n".join(
        [
            f"- Histórico de {fmt_int(len(df))} concursos, validado, direto da API da Caixa.",
            f"- {structure_reading(tests)}",
            f"- {freq}",
            f"- {strategy}",
            f"- {lesson}",
        ]
    )


def summary_table(tests: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Tabela-resumo do topo da seção de resultados."""
    p_freq, p_mc = _p_value(tests, FREQUENCY_TEST), _p_value(tests, FREQUENCY_MC_TEST)
    p_even, p_quad = _p_value(tests, EVEN_ODD_TEST), _p_value(tests, QUADRANT_TEST)
    hot = summary.set_index("estrategia").loc[HOT_STRATEGY]
    min_bonf = summary["p_valor_mc_bonferroni"].min()

    def yes_no(p: float) -> str:
        return "Sim" if p >= ALPHA else "**Não**"

    def wins(p: float) -> str:
        return "**Sim**" if p < ALPHA else "Não"

    return "\n".join(
        [
            "| Pergunta | Resultado | p-valor |",
            "|---|---|---|",
            (
                f"| As 60 dezenas saem com a mesma frequência? | {yes_no(p_freq)} | "
                f"{fmt_p(p_freq)} (qui-quadrado corrigido); {fmt_p(p_mc)} (Monte Carlo) |"
            ),
            f"| A proporção de pares segue a teoria? | {yes_no(p_even)} | {fmt_p(p_even)} |",
            f"| Os quadrantes recebem 25% cada? | {yes_no(p_quad)} | {fmt_p(p_quad)} |",
            (
                f'| Apostar nas "quentes" vence o acaso? | {wins(hot["p_valor_monte_carlo"])} | '
                f"{fmt_p(hot['p_valor_monte_carlo'])} |"
            ),
            (
                f"| Alguma das {len(summary)} estratégias vence o acaso (com Bonferroni)? | "
                f"{wins(min_bonf)} | menor p: {fmt_p(min_bonf)} |"
            ),
        ]
    )


def render_results(df: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Monta a seção de resultados do README (sem os marcadores)."""
    first, last = df.iloc[0], df.iloc[-1]
    return f"""\
Histórico analisado: **{fmt_int(len(df))} concursos**, de {first["data"]:%d/%m/%Y} a \
{last["data"]:%d/%m/%Y} (último concurso: {last["concurso"]}).

{summary_table(tests, summary)}

![Frequência de cada dezena](reports/figures/frequencia_dezenas.png)

{frequency_reading(df, tests)}

{period_reading(tests)}

![Desvio acumulado](reports/figures/frequencia_acumulada.png)

O desvio acumulado mostra como cada dezena se afasta do esperado ao longo do tempo, com \
destaque para as 3 mais e as 3 menos sorteadas. O envelope cinza vale para uma dezena isolada: \
com 60 dezenas, é normal que algumas o ultrapassem.

![Pares e ímpares](reports/figures/pares_impares.png)
![Quadrantes](reports/figures/quadrantes.png)

{structure_reading(tests)}

![Backtest](reports/figures/backtest.png)

{backtest_reading(summary)}

A lição: um teste estatisticamente significativo não implica um efeito explorável. O tamanho \
do efeito importa tanto quanto o p-valor."""


def update_readme(text: str, path: Path = README_FILE) -> None:
    """Substitui o trecho entre os marcadores do README por ``text``.

    Raises:
        ValueError: Se os marcadores não estiverem no README.
    """
    readme = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
    if not pattern.search(readme):
        raise ValueError(f"Marcadores de resultados não encontrados em {path}.")
    block = f"{START_MARKER}\n\n{text}\n\n{END_MARKER}"
    path.write_text(pattern.sub(lambda _: block, readme), encoding="utf-8")


@app.command()
def main(
    input_path: Path = PROCESSED_RESULTS_FILE,
    tests_path: Path = HYPOTHESIS_TESTS_FILE,
    summary_path: Path = BACKTEST_SUMMARY_FILE,
    readme_path: Path = README_FILE,
):
    """Reescreve a seção de resultados do README a partir dos arquivos de resultados."""
    text = render_results(
        load_processed(input_path), pd.read_csv(tests_path), pd.read_csv(summary_path)
    )
    update_readme(text, readme_path)
    logger.success(f"Seção de resultados do README atualizada ({readme_path}).")


if __name__ == "__main__":
    app()
