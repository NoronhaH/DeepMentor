import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Flow verbosity
FLOW_VERBOSITY = True

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Diretório raiz
SRC_DIR = PROJ_ROOT / "deepmentor"

# Estrutura padrão (CCDS)
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Estrutura do projeto CrewAI
FLOWS_DIR = PROJ_ROOT / "flows"

# Caminhos específicos para o flow principal
DEEPMENTOR_FLOW_DIR = FLOWS_DIR / "deepmentor_flow"
TEACHING_CREW_DIR = DEEPMENTOR_FLOW_DIR / "crews" / "teaching_crew"
CREW_CONFIG_DIR = TEACHING_CREW_DIR / "config"
TOOLS_DIR = DEEPMENTOR_FLOW_DIR / "tools"

# Caminhos de memória e conhecimento
KNOWLEDGE_DIR = TEACHING_CREW_DIR / "knowledge"
KNOWLEDGE_DIR_DYN = PROJ_ROOT / "knowledge"
SHORT_TERM_MEMORY_DIR = TEACHING_CREW_DIR / "short_term_memory"
LONG_TERM_MEMORY_DIR = TEACHING_CREW_DIR / "long_term_memory"
ENTITIES_DIR = TEACHING_CREW_DIR / "entities"

# Carregamento de APIs e Segredos
try:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE")

    # OCR NeuralMind
    NEURALMIND_API_KEY = os.environ.get("NEURALMIND_API_KEY")
    NEURALMIND_OCR_URL = os.environ.get("NEURALMIND_OCR_URL")
    
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY não encontrada no .env")
    if not OLLAMA_API_BASE:
        logger.warning("OLLAMA_API_BASE não encontrado no .env. Usando padrão.")
        OLLAMA_API_BASE = "http://localhost:11434"

    logger.info("APIs carregadas do .env")

except Exception as e:
    logger.error(f"Erro ao carregar variáveis de ambiente: {e}")

# OCR NeuralMind
NEURALMIND_API_USE = True

# Models
OLLAMA_MODEL = "ollama/gpt-oss:120b" # (e.g., gpt-oss:120b, gpt-oss:20b)
VISION_MODEL = "qwen3-vl:235b"       #
GPT_MODEL    = "gpt-5-mini"          # (e.g., gpt-5-nano, gpt-5-mini, ...)
LLM_MODEL    = 0 # 0 = OpenAI, 1 = Ollama

# Livros e Materiais (Base de Conhecimentos)
D2L_BOOK_LINK = "https://d2l.ai/d2l-en.pdf"
BOOKS_PATH = PROJ_ROOT / "data" / "raw" / "d2l_book" # Salva livros em data/raw

# Configurações do Flow
OCR_DPI = 300
OCR_PAGES = [
    {
        "section": "summary",
        "start_page": 3,
        "end": 24
    },
    {
        "section": "chapter-3",
        "start_page": 122,
        "end": 164
    }
]

# GRADIO Interface
GRADIO_PUBLIC_SHARE = False

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

logger.info("Configuração do projeto carregada.")