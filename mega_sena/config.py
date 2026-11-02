from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data files
RAW_RESULTS_FILE = RAW_DATA_DIR / "megasena_api.jsonl"
PROCESSED_RESULTS_FILE = PROCESSED_DATA_DIR / "megasena.csv"
FEATURES_FILE = PROCESSED_DATA_DIR / "features.csv"
HYPOTHESIS_TESTS_FILE = REPORTS_DIR / "hypothesis_tests.csv"
BACKTEST_HITS_FILE = MODELS_DIR / "backtest_hits.csv"
BACKTEST_SUMMARY_FILE = MODELS_DIR / "backtest_summary.csv"
BACKTEST_NULL_FILE = MODELS_DIR / "backtest_null_means.csv"

# Mega-Sena
API_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena"
TOTAL_NUMBERS = 60  # dezenas no volante (1 a 60)
NUMBERS_PER_DRAW = 6  # dezenas sorteadas por concurso
DRAW_COLUMNS = [f"d{i}" for i in range(1, NUMBERS_PER_DRAW + 1)]  # dezenas em ordem crescente
ORDER_COLUMNS = [f"bola{i}" for i in range(1, NUMBERS_PER_DRAW + 1)]  # ordem de sorteio

RANDOM_SEED = 42

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
