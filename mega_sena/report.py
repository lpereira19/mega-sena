"""Textos de resultados gerados a partir dos números, para o README e o notebook.

Os números da análise mudam a cada concurso novo. Para os textos que os citam nunca ficarem
desatualizados, eles são montados aqui, inclusive as palavras que dependem do resultado (por
exemplo, "rejeita" ou "não rejeita"). No README, só o trecho entre :data:`START_MARKER` e
:data:`END_MARKER` é reescrito; o restante continua editável à mão.

Uso::

    python -m mega_sena.report   # reescreve a seção de resultados do README
"""

from dataclasses import dataclass
from pathlib import Path
import re

from loguru import logger
import pandas as pd
from scipy.stats import kstest
import typer

from mega_sena import distances
from mega_sena.config import PROJ_ROOT
from mega_sena.dataset import load_processed
from mega_sena.features import draws_matrix, number_frequencies
from mega_sena.games import GAMES, MEGA_SENA, TIMEMANIA, GameSpec
from mega_sena.hypothesis import (
    EVEN_ODD_TEST,
    FREQUENCY_MC_TEST,
    FREQUENCY_TEST,
    OVERLAP_TEST,
    QUADRANT_TEST,
    format_test_name,
)
from mega_sena.modeling.backtest import expected_hits

app = typer.Typer()

README_FILE = PROJ_ROOT / "README.md"
START_MARKER = "<!-- RESULTADOS:INICIO (gerado por mega_sena.report; não edite à mão) -->"
END_MARKER = "<!-- RESULTADOS:FIM -->"
COMPARISON_START = "<!-- COMPARACAO:INICIO (gerado por mega_sena.report; não edite à mão) -->"
COMPARISON_END = "<!-- COMPARACAO:FIM -->"
ALPHA = 0.05
HOT_STRATEGY = "quentes (histórico)"

# O README descreve a Mega-Sena; a comparação entre jogos vem numa etapa posterior.
SPEC = MEGA_SENA
FREQ_TEST = format_test_name(FREQUENCY_TEST, SPEC)
FREQ_MC_TEST = format_test_name(FREQUENCY_MC_TEST, SPEC)
FIGURES = SPEC.paths.figures.relative_to(PROJ_ROOT).as_posix()


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
    rows = tests[tests["teste"].str.startswith(f"{FREQ_TEST}, ")]
    return pd.DataFrame(
        {
            "periodo": rows["teste"].str.removeprefix(f"{FREQ_TEST}, ").str.replace("-", "–"),
            "concursos": rows["observacao"].str.extract(r"(\d+) concursos")[0].astype(int),
            "p_valor": rows["p_valor"],
        }
    ).reset_index(drop=True)


def frequency_reading(df: pd.DataFrame, tests: pd.DataFrame) -> str:
    """Parágrafo sobre a uniformidade das frequências das 60 dezenas."""
    freq = number_frequencies(draws_matrix(df))
    expected = len(df) * SPEC.sorteadas / SPEC.n
    p_freq, p_mc = _p_value(tests, FREQ_TEST), _p_value(tests, FREQ_MC_TEST)
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
        f"| *esperado ao acaso* | *{fmt_num(expected_hits(SPEC))}* | | |",
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
    p_freq = _p_value(tests, FREQ_TEST)
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
    p_freq, p_mc = _p_value(tests, FREQ_TEST), _p_value(tests, FREQ_MC_TEST)
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


def distance_reading(df: pd.DataFrame, distance_tests: pd.DataFrame) -> str:
    """Leitura da análise de distância entre as dezenas de um mesmo concurso."""
    draws = draws_matrix(df)
    n = len(draws)
    with_pair = (distances.consecutive_pairs(draws) > 0).mean()
    expected_pair = 1 - distances.consecutive_pairs_pmf()[0]
    runs = distances.longest_run(draws)
    run_pmf = distances.longest_run_pmf()
    smallest = distances.min_gap(draws).mean()
    min_pmf = distances.min_gap_pmf()
    expected_smallest = float((min_pmf * min_pmf.index).sum())

    facts = (
        f"- **{fmt_num(with_pair * 100, 1)}%** dos concursos tiveram ao menos um par de dezenas "
        f"consecutivas (como 22 e 23), contra {fmt_num(expected_pair * 100, 1)}% esperados.\n"
        f"- **{fmt_int(int((runs >= 3).sum()))}** concursos tiveram 3 ou mais dezenas seguidas "
        f"(esperado: {fmt_num(run_pmf.loc[3:].sum() * n, 0)}), e "
        f"**{fmt_int(int((runs >= 4).sum()))}** tiveram 4 ou mais "
        f"(esperado: {fmt_num(run_pmf.loc[4:].sum() * n, 1)}).\n"
        f"- A menor distância entre duas dezenas de um concurso é, em média, "
        f"**{fmt_num(smallest, 2)}**, contra {fmt_num(expected_smallest, 2)} esperada."
    )

    lines = ["| Teste | p-valor | p (Bonferroni) |", "|---|---|---|"]
    lines += [
        f"| {row.teste} | {fmt_p(row.p_valor)} | {fmt_p(row.p_valor_bonferroni)} |"
        for row in distance_tests.itertuples()
    ]

    significant = distance_tests[distance_tests["p_valor_bonferroni"] < ALPHA]
    nominal = distance_tests[
        (distance_tests["p_valor"] < ALPHA) & (distance_tests["p_valor_bonferroni"] >= ALPHA)
    ]
    if significant.empty:
        verdict = "**Nenhuma medida de distância foge do acaso** depois da correção para múltiplas comparações."
        if not nominal.empty:
            names = _join([f'"{name}"' for name in nominal["teste"]])
            verdict += (
                f" {names} tem p < 0,05 isoladamente, mas foram feitos {len(distance_tests)} "
                "testes, e com Bonferroni o resultado deixa de ser significativo."
            )
    else:
        names = _join([f'"{name}"' for name in significant["teste"]])
        verdict = (
            f"**{names} foge do esperado mesmo após a correção de Bonferroni.** Vale investigar "
            "o tamanho do efeito antes de qualquer conclusão."
        )

    nearest = distances.nearest_by_number(draws)
    outliers = nearest.index[nearest["z"].abs() > 2].tolist()
    if outliers:
        verdict += (
            f" No gráfico por dezena, {_join([f'{x:02d}' for x in outliers])} "
            f"{'ficam' if len(outliers) > 1 else 'fica'} fora da faixa de 95%; com 60 dezenas, "
            "espera-se que cerca de 3 fiquem fora por puro acaso."
        )
    return f"{facts}\n\n" + "\n".join(lines) + f"\n\n{verdict}"


def render_results(
    df: pd.DataFrame,
    tests: pd.DataFrame,
    summary: pd.DataFrame,
    distance_tests: pd.DataFrame,
) -> str:
    """Monta a seção de resultados do README (sem os marcadores)."""
    first, last = df.iloc[0], df.iloc[-1]
    return f"""\
Histórico analisado: **{fmt_int(len(df))} concursos**, de {first["data"]:%d/%m/%Y} a \
{last["data"]:%d/%m/%Y} (último concurso: {last["concurso"]}).

{summary_table(tests, summary)}

![Frequência de cada dezena]({FIGURES}/frequencia_dezenas.png)

{frequency_reading(df, tests)}

{period_reading(tests)}

![Desvio acumulado]({FIGURES}/frequencia_acumulada.png)

O desvio acumulado mostra como cada dezena se afasta do esperado ao longo do tempo, com \
destaque para as 3 mais e as 3 menos sorteadas. O envelope cinza vale para uma dezena isolada: \
com 60 dezenas, é normal que algumas o ultrapassem.

![Pares e ímpares]({FIGURES}/pares_impares.png)
![Quadrantes]({FIGURES}/quadrantes.png)

{structure_reading(tests)}

### Distância entre as dezenas

Quão próximas saem as dezenas de um mesmo concurso? Números "colados", sequências e a dezena \
mais próxima de cada uma, comparados com as distribuições exatas de um sorteio aleatório.

![Distâncias]({FIGURES}/distancias.png)

{distance_reading(df, distance_tests)}

![Vizinhança]({FIGURES}/vizinhanca.png)

À esquerda, distâncias curtas são mais comuns só porque existem mais pares assim no volante \
(há 59 pares a distância 1 e só 1 par a distância 59). À direita, as dezenas das pontas (01 e \
60) têm a vizinha mais longe porque só têm vizinhas de um lado.

### Backtest

![Backtest]({FIGURES}/backtest.png)

{backtest_reading(summary)}

A lição: um teste estatisticamente significativo não implica um efeito explorável. O tamanho \
do efeito importa tanto quanto o p-valor."""


# ---------------------------------------------------------------------------------------------
# Comparação entre as loterias
# ---------------------------------------------------------------------------------------------


@dataclass
class GameResults:
    """Resultados salvos de um jogo, lidos dos arquivos do pipeline."""

    spec: GameSpec
    df: pd.DataFrame
    tests: pd.DataFrame
    distance_tests: pd.DataFrame
    summary: pd.DataFrame


def load_game_results(spec: GameSpec) -> GameResults | None:
    """Lê os resultados de um jogo, ou ``None`` se o pipeline ainda não rodou para ele."""
    paths = spec.paths
    files = [paths.processed, paths.hypothesis_tests, paths.distance_tests, paths.backtest_summary]
    if not all(f.exists() for f in files):
        return None
    return GameResults(
        spec,
        load_processed(spec),
        pd.read_csv(paths.hypothesis_tests),
        pd.read_csv(paths.distance_tests),
        pd.read_csv(paths.backtest_summary),
    )


def game_rule(spec: GameSpec) -> str:
    """Regra do jogo em poucas palavras: ``"6 de 60"``, ``"20 de 100, aposta 50"``..."""
    rule = f"{spec.sorteadas} de {spec.n}"
    if spec.aposta != spec.sorteadas:
        rule += f", aposta {spec.aposta}"
    if spec.sorteios_por_concurso > 1:
        rule += f", {spec.sorteios_por_concurso} sorteios"
    if spec.extra_sorteadas > 1:
        rule += f" + {spec.extra_sorteadas} de {spec.extra_total} {spec.extra_nome.lower()}"
    elif spec.extra:
        rule += f" + {spec.extra_nome}"
    return rule


def _own_tests(results: GameResults) -> pd.DataFrame:
    """Testes que só existem em alguns jogos: sorteio extra e os dois sorteios da Dupla Sena."""
    names = results.tests["teste"]
    spec = results.spec
    own = names.eq(OVERLAP_TEST)
    if spec.extra:
        own |= names.str.startswith(f"{spec.extra_nome}:")
    return results.tests[own]


def _is_exploratory(tests: pd.DataFrame, spec: GameSpec) -> pd.Series:
    """Linhas que não entram na contagem de testes: recortes por período e a versão Monte Carlo
    do teste de frequência (repete a pergunta do teste corrigido)."""
    return tests["observacao"].fillna("").str.startswith("pós-hoc") | tests["teste"].eq(
        format_test_name(FREQUENCY_MC_TEST, spec)
    )


def confirmatory_p_values(results: list[GameResults]) -> pd.DataFrame:
    """Todos os testes confirmatórios de todos os jogos, com o p-valor sem correção.

    Entram os testes de hipótese (menos os exploratórios), os de distância e o p-valor de
    Monte Carlo de cada estratégia do backtest.
    """
    frames = []
    for r in results:
        tests = r.tests[~_is_exploratory(r.tests, r.spec)]
        backtest = pd.DataFrame(
            {
                "teste": "Backtest: " + r.summary["estrategia"],
                "p_valor": r.summary["p_valor_monte_carlo"],
            }
        )
        for part in [
            tests[["teste", "p_valor"]],
            r.distance_tests[["teste", "p_valor"]],
            backtest,
        ]:
            frames.append(part.assign(jogo=r.spec.nome))
    return pd.concat(frames, ignore_index=True).dropna(subset=["p_valor"])


def _cell(p: float) -> str:
    """P-valor de célula de tabela, em negrito quando abaixo de 0,05."""
    return f"**{fmt_p(p)}**" if p < ALPHA else fmt_p(p)


def comparison_table(results: list[GameResults]) -> str:
    """Uma linha por jogo com os p-valores dos testes comuns a todos."""
    lines = [
        (
            "| Jogo | Regra | Concursos | Frequência | Pares/ímpares | Quadrantes "
            "| Distâncias¹ | Teste próprio² | Estratégias³ |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        spec = r.spec
        own = _own_tests(r)
        own_cell = " / ".join(_cell(p) for p in own["p_valor"]) if len(own) else "–"
        lines.append(
            f"| {spec.nome} | {game_rule(spec)} "
            f"| {fmt_int(r.df['concurso'].nunique())} (desde {r.df['data'].min():%Y}) "
            f"| {_cell(_p_value(r.tests, format_test_name(FREQUENCY_TEST, spec)))} "
            f"| {_cell(_p_value(r.tests, EVEN_ODD_TEST))} "
            f"| {_cell(_p_value(r.tests, QUADRANT_TEST))} "
            f"| {_cell(r.distance_tests['p_valor_bonferroni'].min())} "
            f"| {own_cell} "
            f"| {_cell(r.summary['p_valor_mc_bonferroni'].min())} |"
        )
    return "\n".join(lines)


def comparison_reading(results: list[GameResults]) -> str:
    """Leitura da tabela levando em conta quantos testes foram feitos ao todo."""
    p_values = confirmatory_p_values(results)
    total = len(p_values)
    significant = p_values[p_values["p_valor"] < ALPHA]
    threshold = ALPHA / total
    survivors = p_values[p_values["p_valor"] < threshold].sort_values("p_valor")
    uniformity = kstest(p_values["p_valor"], "uniform").pvalue

    # Estratégias de backtest que dão significativas num jogo cuja frequência já deu: as
    # "quentes" e "frias" são escolhidas pela frequência, então repetem o mesmo achado.
    biased_games = {
        r.spec.nome
        for r in results
        if _p_value(r.tests, format_test_name(FREQUENCY_TEST, r.spec)) < ALPHA
    }
    echoes = significant["teste"].str.startswith("Backtest:") & significant["jogo"].isin(
        biased_games
    )

    text = (
        f"Ao todo são **{total} testes** nos {len(results)} jogos (sem contar os recortes por "
        f"período, que são exploratórios). Se todos os sorteios fossem perfeitamente "
        f"aleatórios, cerca de {fmt_num(ALPHA * total, 0)} dariam p < 0,05 só por acaso; deram "
        f"**{len(significant)}**."
    )
    if echoes.any():
        text += (
            " Parte do excesso é o mesmo achado contado mais de uma vez: "
            + (
                f"{int(echoes.sum())} deles são estratégias"
                if echoes.sum() > 1
                else "um deles é uma estratégia"
            )
            + " do backtest de jogos cuja frequência já deu significativa, e essas estratégias "
            "escolhem as dezenas justamente pela frequência."
        )
    text += (
        f" Olhados juntos, os {total} p-valores são "
        f"{'compatíveis' if uniformity >= ALPHA else 'incompatíveis'} com a distribuição "
        f"uniforme esperada sob H0 (Kolmogorov-Smirnov, {p_eq(uniformity)}; aproximado, porque "
        f"testes do mesmo jogo usam os mesmos sorteios)."
    )
    if survivors.empty:
        return (
            f"{text}\n\nCom a correção de Bonferroni para os {total} testes (p < "
            f"{fmt_num(threshold, 4)}), **nenhum resultado sobrevive**: o conjunto é compatível "
            f"com sorteios aleatórios em todas as loterias."
        )

    found = _join(
        [
            f"o teste de {row.teste[0].lower()}{row.teste[1:]} da {row.jogo} ({p_eq(row.p_valor)})"
            for row in survivors.itertuples()
        ]
    )
    plural = len(survivors) > 1
    text += (
        f"\n\nCom a correção de Bonferroni para os {total} testes (p < {fmt_num(threshold, 4)}), "
        f"{'sobrevivem' if plural else 'sobrevive'} só {found}."
    )
    by_name = {r.spec.nome: r for r in results}
    for jogo in survivors["jogo"].unique():
        best = by_name[jogo].summary["p_valor_mc_bonferroni"].min()
        if best >= ALPHA:
            text += (
                f" Mesmo assim, nenhuma estratégia de aposta da {jogo} vence o acaso no backtest "
                f"(menor p com Bonferroni: {fmt_p(best)}): o desvio é estatisticamente "
                f"detectável, mas pequeno demais para ser explorado."
            )
    mega = by_name.get(MEGA_SENA.nome)
    if mega is not None:
        p_mega = _p_value(mega.tests, FREQ_TEST)
        if p_mega < ALPHA and p_mega >= threshold:
            text += (
                f" Já o desvio de frequência da Mega-Sena ({p_eq(p_mega)}), destaque da seção "
                f"anterior, não passa nessa régua: olhando {len(results)} loterias, um p desse "
                f"tamanho em algum lugar é esperado."
            )
    return text


def render_comparison(results: list[GameResults]) -> str:
    """Monta a seção de comparação entre as loterias (sem os marcadores)."""
    return f"""\
{comparison_table(results)}

¹ Menor p-valor entre os testes de distância do jogo, já corrigido por Bonferroni dentro do \
jogo. ² Mês da Sorte (Dia de Sorte), Time do Coração (Timemania, só até o concurso \
{TIMEMANIA.extra_ate}, enquanto a lista de clubes foi fixa), trevos por trevo e por combinação \
(+Milionária) e dezenas em comum entre os dois sorteios (Dupla Sena). ³ Menor p-valor entre as \
estratégias do backtest, corrigido por Bonferroni. Em negrito, p < 0,05.

{comparison_reading(results)}

As figuras e tabelas de cada jogo ficam em `reports/figures/<jogo>` e `reports/<jogo>`."""


def update_readme(
    text: str,
    path: Path = README_FILE,
    start: str = START_MARKER,
    end: str = END_MARKER,
) -> None:
    """Substitui o trecho entre os marcadores ``start`` e ``end`` do README por ``text``.

    Raises:
        ValueError: Se os marcadores não estiverem no README.
    """
    readme = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(readme):
        raise ValueError(f"Marcadores {start!r} não encontrados em {path}.")
    block = f"{start}\n\n{text}\n\n{end}"
    path.write_text(pattern.sub(lambda _: block, readme), encoding="utf-8")


def run_comparison(readme_path: Path = README_FILE) -> None:
    """Reescreve a comparação entre as loterias com os jogos que já têm resultados."""
    results = [r for spec in GAMES.values() if (r := load_game_results(spec)) is not None]
    if not results:
        logger.warning("Nenhum jogo com resultados; comparação não atualizada.")
        return
    update_readme(render_comparison(results), readme_path, COMPARISON_START, COMPARISON_END)
    logger.success(f"Comparação entre {len(results)} loterias atualizada ({readme_path}).")


@app.command()
def main():
    """Reescreve as seções geradas do README a partir dos arquivos de resultados."""
    run()
    run_comparison()


def run(readme_path: Path = README_FILE) -> None:
    """Reescreve a seção de resultados do README com os resultados atuais da Mega-Sena."""
    paths = SPEC.paths
    text = render_results(
        load_processed(SPEC),
        pd.read_csv(paths.hypothesis_tests),
        pd.read_csv(paths.backtest_summary),
        pd.read_csv(paths.distance_tests),
    )
    update_readme(text, readme_path)
    logger.success(f"Seção de resultados do README atualizada ({readme_path}).")


if __name__ == "__main__":
    app()
