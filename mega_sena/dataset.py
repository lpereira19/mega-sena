"""ETL dos resultados da Mega-Sena.

Fonte principal: API pública do Portal de Loterias da Caixa, que expõe o último concurso em
``{API_URL}/`` e cada concurso em ``{API_URL}/{numero}``. Como fonte alternativa, aceita um
arquivo CSV/XLSX baixado manualmente do portal (ver :func:`load_results_file`).

Etapas:

1. **Extract**: download incremental dos JSONs para ``data/raw``. O arquivo bruto só recebe
   novas linhas, nunca é editado, e uma nova execução baixa apenas os concursos que faltam.
2. **Transform**: parsing, tipagem (datas, inteiros), ordenação das dezenas e validação.
3. **Load**: CSV limpo, um concurso por linha, em ``data/processed``.

Uso::

    python -m mega_sena.dataset                  # baixa o que falta e processa
    python -m mega_sena.dataset --skip-download  # só reprocessa o arquivo bruto
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time

from loguru import logger
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
import typer
from urllib3.util.retry import Retry

from mega_sena.config import (
    API_URL,
    DRAW_COLUMNS,
    NUMBERS_PER_DRAW,
    ORDER_COLUMNS,
    PROCESSED_RESULTS_FILE,
    RAW_RESULTS_FILE,
    TOTAL_NUMBERS,
)

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


def fetch_contest(number: int | None = None, timeout: float = 30.0, delay: float = 0.0) -> dict:
    """Baixa o JSON de um concurso na API da Caixa.

    Args:
        number: Número do concurso. ``None`` retorna o concurso mais recente.
        timeout: Tempo máximo de espera por requisição, em segundos.
        delay: Pausa antes da requisição, em segundos, para respeitar o limite da API.

    Returns:
        O JSON do concurso exatamente como a API devolve.

    Raises:
        requests.HTTPError: Se a API responder com erro depois das tentativas de retry.
    """
    url = f"{API_URL}/{number}" if number is not None else f"{API_URL}/"
    time.sleep(delay)
    response = _get_session().get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def read_raw(path: Path = RAW_RESULTS_FILE) -> list[dict]:
    """Lê o arquivo bruto (JSON Lines, um concurso por linha). Retorna ``[]`` se não existir."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def download_history(
    raw_path: Path = RAW_RESULTS_FILE, max_workers: int = 2, delay: float = 0.25
) -> list[dict]:
    """Baixa de forma incremental os concursos que ainda não estão no arquivo bruto.

    Os registros novos são acrescentados ao final de ``raw_path``; nada já salvo é reescrito.
    Concursos que falharem são registrados no log e serão buscados na próxima execução.

    Args:
        raw_path: Arquivo JSON Lines com as respostas brutas da API.
        max_workers: Número de requisições em paralelo. A API bloqueia rajadas (HTTP 403),
            então valores altos tendem a ser mais lentos, não mais rápidos.
        delay: Pausa antes de cada requisição, em segundos.

    Returns:
        Todos os registros brutos disponíveis (antigos + novos).
    """
    records = read_raw(raw_path)
    downloaded = {record["numero"] for record in records}
    latest = fetch_contest()["numero"]
    missing = sorted(set(range(1, latest + 1)) - downloaded)

    if not missing:
        logger.info(f"Arquivo bruto já está atualizado até o concurso {latest}.")
        return records

    logger.info(f"Baixando {len(missing)} concursos (último: {latest})...")
    new_records, failures = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_contest, number, delay=delay): number for number in missing}
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


def load_results_file(path: Path) -> pd.DataFrame:
    """Carrega um arquivo CSV/XLSX de resultados baixado manualmente do portal da Caixa.

    O layout esperado é o da planilha oficial: uma coluna ``Concurso``, uma coluna de data
    (``Data do Sorteio``) e seis colunas de dezenas (``Bola1`` ... ``Bola6``). A leitura de
    XLSX requer o pacote ``openpyxl``.

    Args:
        path: Caminho do arquivo ``.csv`` ou ``.xlsx``.

    Returns:
        DataFrame validado, no mesmo formato de :func:`build_dataframe`, sem as colunas
        financeiras.
    """
    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    columns = {col: str(col).strip().lower() for col in raw.columns}
    contest_col = next(c for c, name in columns.items() if name.startswith("concurso"))
    date_col = next(c for c, name in columns.items() if name.startswith("data"))
    ball_cols = [c for c, name in columns.items() if name.startswith(("bola", "dezena"))]
    if len(ball_cols) != NUMBERS_PER_DRAW:
        raise ValueError(f"Esperadas {NUMBERS_PER_DRAW} colunas de dezenas, achei {ball_cols}.")

    balls = raw[ball_cols].astype(int).to_numpy()
    df = pd.DataFrame(
        {
            "concurso": raw[contest_col].astype(int),
            "data": pd.to_datetime(raw[date_col], dayfirst=True),
        }
    )
    df[DRAW_COLUMNS] = pd.DataFrame(balls).apply(sorted, axis=1, result_type="expand").to_numpy()
    df[ORDER_COLUMNS] = balls
    df = df.sort_values("concurso").reset_index(drop=True)
    validate_results(df)
    return df


# ---------------------------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------------------------


def parse_contest(record: dict) -> dict:
    """Converte o JSON bruto de um concurso em um registro tabular plano.

    Args:
        record: Resposta da API para um concurso.

    Returns:
        Dicionário com ``concurso``, ``data``, as dezenas em ordem crescente (``d1``..``d6``),
        as dezenas na ordem de sorteio (``bola1``..``bola6``), o número de ganhadores da sena,
        se o prêmio acumulou e o valor arrecadado.
    """
    sorted_numbers = sorted(int(n) for n in record["listaDezenas"])
    draw_order = [int(n) for n in record.get("dezenasSorteadasOrdemSorteio") or []]
    jackpot = next(
        (tier for tier in record.get("listaRateioPremio") or [] if tier.get("faixa") == 1), {}
    )
    return {
        "concurso": int(record["numero"]),
        "data": pd.to_datetime(record["dataApuracao"], format="%d/%m/%Y"),
        **dict(zip(DRAW_COLUMNS, sorted_numbers)),
        **dict(zip(ORDER_COLUMNS, draw_order or sorted_numbers)),
        "ganhadores_sena": int(jackpot.get("numeroDeGanhadores", 0)),
        "acumulado": bool(record.get("acumulado")),
        "valor_arrecadado": float(record.get("valorArrecadado") or 0.0),
    }


def validate_results(df: pd.DataFrame) -> None:
    """Checa a integridade do conjunto de resultados.

    Raises:
        ValueError: Se houver concurso duplicado, dezena fora de 1..60 ou dezena repetida
            dentro de um concurso.
    """
    draws = df[DRAW_COLUMNS].to_numpy()
    if df["concurso"].duplicated().any():
        raise ValueError("Há concursos duplicados.")
    if ((draws < 1) | (draws > TOTAL_NUMBERS)).any():
        raise ValueError(f"Há dezenas fora do intervalo 1..{TOTAL_NUMBERS}.")
    if (pd.DataFrame(draws).nunique(axis=1) != NUMBERS_PER_DRAW).any():
        raise ValueError("Há concursos com dezenas repetidas.")

    gaps = sorted(set(range(1, df["concurso"].max() + 1)) - set(df["concurso"]))
    if gaps:
        logger.warning(f"{len(gaps)} concursos ausentes na sequência (ex.: {gaps[:5]}).")


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Transforma a lista de JSONs brutos em um DataFrame limpo e validado.

    Args:
        records: Respostas brutas da API (ver :func:`download_history`).

    Returns:
        Um concurso por linha, ordenado pelo número do concurso.
    """
    df = (
        pd.DataFrame([parse_contest(record) for record in records])
        .drop_duplicates("concurso", keep="last")
        .sort_values("concurso")
        .reset_index(drop=True)
    )
    validate_results(df)
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Pivota o formato largo (6 colunas de dezenas) para o formato longo.

    Returns:
        DataFrame com uma linha por dezena sorteada: ``concurso``, ``data``, ``posicao`` (1 a 6,
        na ordem de sorteio) e ``dezena``.
    """
    long = df.melt(
        id_vars=["concurso", "data"],
        value_vars=ORDER_COLUMNS,
        var_name="posicao",
        value_name="dezena",
    )
    long["posicao"] = long["posicao"].str.removeprefix("bola").astype(int)
    return long.sort_values(["concurso", "posicao"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------------------------


def load_processed(path: Path = PROCESSED_RESULTS_FILE) -> pd.DataFrame:
    """Lê o CSV processado, com a coluna ``data`` já convertida para datetime."""
    return pd.read_csv(path, parse_dates=["data"])


@app.command()
def main(
    raw_path: Path = RAW_RESULTS_FILE,
    output_path: Path = PROCESSED_RESULTS_FILE,
    skip_download: bool = typer.Option(False, help="Só reprocessa o arquivo bruto existente."),
    workers: int = typer.Option(2, help="Requisições paralelas à API."),
    delay: float = typer.Option(0.25, help="Pausa antes de cada requisição, em segundos."),
):
    """Executa o ETL completo: download incremental, transformação e gravação do CSV."""
    records = read_raw(raw_path) if skip_download else download_history(raw_path, workers, delay)
    if not records:
        raise typer.Exit("Nenhum dado bruto disponível. Rode sem --skip-download.")

    logger.info("Transformando dados brutos...")
    df = build_dataframe(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(
        f"{len(df)} concursos ({df['data'].min():%d/%m/%Y} a {df['data'].max():%d/%m/%Y}) "
        f"salvos em {output_path}."
    )


if __name__ == "__main__":
    app()
