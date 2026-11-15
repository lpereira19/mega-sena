"""ETL dos resultados das loterias da Caixa.

Fonte principal: API pública do Portal de Loterias da Caixa, que expõe o último concurso de um
jogo em ``{api_url}/`` e cada concurso em ``{api_url}/{numero}`` (ver
:attr:`mega_sena.games.GameSpec.api_url`). Como fonte alternativa, aceita um arquivo CSV/XLSX
baixado manualmente do portal (ver :func:`load_results_file`).

Etapas:

1. **Extract**: download incremental dos JSONs para ``data/raw``. O arquivo bruto só recebe
   novas linhas, nunca é editado, e uma nova execução baixa apenas os concursos que faltam.
2. **Transform**: parsing, tipagem (datas, inteiros), ordenação das dezenas e validação.
3. **Load**: CSV limpo, um concurso por linha, em ``data/processed``.

Uso::

    python -m mega_sena.dataset                  # baixa o que falta e processa (Mega-Sena)
    python -m mega_sena.dataset --jogo megasena  # escolhe o jogo
    python -m mega_sena.dataset --skip-download  # só reprocessa o arquivo bruto
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import threading
import time

from loguru import logger
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
import typer
from urllib3.util.retry import Retry

from mega_sena.games import MEGA_SENA, GameSpec, get_game

app = typer.Typer()

_thread_local = threading.local()


# ---------------------------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------------------------


def _get_session() -> requests.Session:
    """Retorna uma sessão HTTP por thread, com retry exponencial para erros transitórios."""
    if not hasattr(_thread_local, "session"):
        # O firewall da API responde 403 quando recebe requisições demais; tratamos como
        # transitório, assim como 429 e 5xx.
        retry = Retry(total=6, backoff_factor=2.0, status_forcelist=(403, 429, 500, 502, 503, 504))
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers["User-Agent"] = "Mozilla/5.0 (mega-sena-data-science)"
        _thread_local.session = session
    return _thread_local.session


def fetch_contest(
    spec: GameSpec, number: int | None = None, timeout: float = 30.0, delay: float = 0.0
) -> dict:
    """Baixa o JSON de um concurso na API da Caixa.

    Args:
        spec: Jogo a consultar.
        number: Número do concurso. ``None`` retorna o concurso mais recente.
        timeout: Tempo máximo de espera por requisição, em segundos.
        delay: Pausa antes da requisição, em segundos, para respeitar o limite da API.

    Returns:
        O JSON do concurso exatamente como a API devolve.

    Raises:
        requests.HTTPError: Se a API responder com erro depois das tentativas de retry.
    """
    url = f"{spec.api_url}/{number}" if number is not None else f"{spec.api_url}/"
    time.sleep(delay)
    response = _get_session().get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def read_raw(path: Path) -> list[dict]:
    """Lê o arquivo bruto (JSON Lines, um concurso por linha). Retorna ``[]`` se não existir."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def download_history(spec: GameSpec, max_workers: int = 2, delay: float = 0.25) -> list[dict]:
    """Baixa de forma incremental os concursos que ainda não estão no arquivo bruto do jogo.

    Os registros novos são acrescentados ao final do arquivo bruto; nada já salvo é reescrito.
    Concursos que falharem são registrados no log e serão buscados na próxima execução.

    Args:
        spec: Jogo a baixar.
        max_workers: Número de requisições em paralelo. A API bloqueia rajadas (HTTP 403),
            então valores altos tendem a ser mais lentos, não mais rápidos.
        delay: Pausa antes de cada requisição, em segundos.

    Returns:
        Todos os registros brutos disponíveis (antigos + novos).
    """
    raw_path = spec.paths.raw
    records = read_raw(raw_path)
    downloaded = {record["numero"] for record in records}
    latest = fetch_contest(spec)["numero"]
    missing = sorted(set(range(1, latest + 1)) - downloaded)

    if not missing:
        logger.info(f"{spec.nome}: arquivo bruto já está atualizado até o concurso {latest}.")
        return records

    logger.info(f"{spec.nome}: baixando {len(missing)} concursos (último: {latest})...")
    new_records, failures = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_contest, spec, number, delay=delay): number for number in missing
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                new_records.append(future.result())
            except requests.RequestException as exc:
                failures.append(futures[future])
                logger.warning(f"Concurso {futures[future]} falhou: {exc}")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as f:
        for record in sorted(new_records, key=lambda r: r["numero"]):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if failures:
        logger.warning(f"{len(failures)} concursos falharam; rode novamente para completar.")
    return records + new_records


def load_results_file(path: Path, spec: GameSpec = MEGA_SENA) -> pd.DataFrame:
    """Carrega um arquivo CSV/XLSX de resultados baixado manualmente do portal da Caixa.

    O layout esperado é o da planilha oficial: uma coluna ``Concurso``, uma coluna de data
    (``Data do Sorteio``) e uma coluna por dezena sorteada (``Bola1`` ... ``BolaK``). A leitura
    de XLSX requer o pacote ``openpyxl``.

    Args:
        path: Caminho do arquivo ``.csv`` ou ``.xlsx``.
        spec: Jogo dos resultados.

    Returns:
        DataFrame validado, no mesmo formato de :func:`build_dataframe`, sem as colunas
        financeiras.
    """
    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    columns = {col: str(col).strip().lower() for col in raw.columns}
    contest_col = next(c for c, name in columns.items() if name.startswith("concurso"))
    date_col = next(c for c, name in columns.items() if name.startswith("data"))
    ball_cols = [c for c, name in columns.items() if name.startswith(("bola", "dezena"))]
    if len(ball_cols) != spec.sorteadas:
        raise ValueError(f"Esperadas {spec.sorteadas} colunas de dezenas, achei {ball_cols}.")

    balls = raw[ball_cols].astype(int).to_numpy()
    df = pd.DataFrame(
        {
            "concurso": raw[contest_col].astype(int),
            "data": pd.to_datetime(raw[date_col], dayfirst=True),
        }
    )
    df[spec.draw_columns] = (
        pd.DataFrame(balls).apply(sorted, axis=1, result_type="expand").to_numpy()
    )
    df[spec.order_columns] = balls
    df = df.sort_values("concurso").reset_index(drop=True)
    validate_results(df, spec)
    return df


# ---------------------------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------------------------


def parse_contest(record: dict, spec: GameSpec = MEGA_SENA) -> dict:
    """Converte o JSON bruto de um concurso em um registro tabular plano.

    Args:
        record: Resposta da API para um concurso.
        spec: Jogo do concurso.

    Returns:
        Dicionário com ``concurso``, ``data``, as dezenas em ordem crescente (``d1``..``dK``),
        as dezenas na ordem de sorteio (``bola1``..``bolaK``), o número de ganhadores do prêmio
        máximo, se o prêmio acumulou e o valor arrecadado.
    """
    sorted_numbers = sorted(int(n) for n in record["listaDezenas"])
    draw_order = [int(n) for n in record.get("dezenasSorteadasOrdemSorteio") or []]
    jackpot = next(
        (tier for tier in record.get("listaRateioPremio") or [] if tier.get("faixa") == 1), {}
    )
    return {
        "concurso": int(record["numero"]),
        "data": pd.to_datetime(record["dataApuracao"], format="%d/%m/%Y"),
        **dict(zip(spec.draw_columns, sorted_numbers)),
        **dict(zip(spec.order_columns, draw_order[: spec.sorteadas] or sorted_numbers)),
        "ganhadores_premio_maximo": int(jackpot.get("numeroDeGanhadores", 0)),
        "acumulado": bool(record.get("acumulado")),
        "valor_arrecadado": float(record.get("valorArrecadado") or 0.0),
    }


def validate_results(df: pd.DataFrame, spec: GameSpec = MEGA_SENA) -> None:
    """Checa a integridade do conjunto de resultados.

    Raises:
        ValueError: Se houver concurso duplicado, dezena fora do volante ou dezena repetida
            dentro de um concurso.
    """
    draws = df[spec.draw_columns].to_numpy()
    if df["concurso"].duplicated().any():
        raise ValueError("Há concursos duplicados.")
    if ((draws < spec.menor) | (draws > spec.maior)).any():
        raise ValueError(f"Há dezenas fora do intervalo {spec.menor}..{spec.maior}.")
    if (pd.DataFrame(draws).nunique(axis=1) != spec.sorteadas).any():
        raise ValueError("Há concursos com dezenas repetidas.")

    gaps = sorted(set(range(1, df["concurso"].max() + 1)) - set(df["concurso"]))
    if gaps:
        logger.warning(f"{len(gaps)} concursos ausentes na sequência (ex.: {gaps[:5]}).")


def build_dataframe(records: list[dict], spec: GameSpec = MEGA_SENA) -> pd.DataFrame:
    """Transforma a lista de JSONs brutos em um DataFrame limpo e validado.

    Args:
        records: Respostas brutas da API (ver :func:`download_history`).
        spec: Jogo dos registros.

    Returns:
        Um concurso por linha, ordenado pelo número do concurso.
    """
    df = (
        pd.DataFrame([parse_contest(record, spec) for record in records])
        .drop_duplicates("concurso", keep="last")
        .sort_values("concurso")
        .reset_index(drop=True)
    )
    validate_results(df, spec)
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Pivota o formato largo (uma coluna por dezena) para o formato longo.

    Returns:
        DataFrame com uma linha por dezena sorteada: ``concurso``, ``data``, ``posicao`` (na
        ordem de sorteio) e ``dezena``.
    """
    order_columns = [col for col in df.columns if re.fullmatch(r"bola\d+", col)]
    long = df.melt(
        id_vars=["concurso", "data"],
        value_vars=order_columns,
        var_name="posicao",
        value_name="dezena",
    )
    long["posicao"] = long["posicao"].str.removeprefix("bola").astype(int)
    return long.sort_values(["concurso", "posicao"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------------------------


def load_processed(spec: GameSpec = MEGA_SENA) -> pd.DataFrame:
    """Lê o CSV processado do jogo, com a coluna ``data`` já convertida para datetime."""
    return pd.read_csv(spec.paths.processed, parse_dates=["data"])


def run(
    spec: GameSpec, skip_download: bool = False, workers: int = 2, delay: float = 0.25
) -> pd.DataFrame:
    """Executa o ETL completo de um jogo e grava o CSV processado.

    Raises:
        FileNotFoundError: Se não houver dados brutos e o download estiver desligado.
    """
    records = read_raw(spec.paths.raw) if skip_download else download_history(spec, workers, delay)
    if not records:
        raise FileNotFoundError(f"{spec.nome}: nenhum dado bruto. Rode sem --skip-download.")

    logger.info(f"{spec.nome}: transformando dados brutos...")
    df = build_dataframe(records, spec)

    output_path = spec.paths.processed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(
        f"{len(df)} concursos ({df['data'].min():%d/%m/%Y} a {df['data'].max():%d/%m/%Y}) "
        f"salvos em {output_path}."
    )
    return df


@app.command()
def main(
    jogo: str = typer.Option(MEGA_SENA.slug, help="Jogo (slug da API)."),
    skip_download: bool = typer.Option(False, help="Só reprocessa o arquivo bruto existente."),
    workers: int = typer.Option(2, help="Requisições paralelas à API."),
    delay: float = typer.Option(0.25, help="Pausa antes de cada requisição, em segundos."),
):
    """Executa o ETL completo: download incremental, transformação e gravação do CSV."""
    run(get_game(jogo), skip_download, workers, delay)


if __name__ == "__main__":
    app()
