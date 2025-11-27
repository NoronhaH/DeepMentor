# Execução
# python3 -m deepmentor.data_loader download-book
# python3 -m deepmentor.data_loader prepare-images
# python3 -m deepmentor.data_loader run-ocr
# python3 -m deepmentor.data_loader calculate-tokens

from PIL import Image
from IPython.display import display
import typer
import requests
import sys
import warnings
import json
import tiktoken
from pathlib import Path
from tqdm import tqdm
from PyPDF2 import PdfReader
from pdf2image import convert_from_path

# Ignora warnings para um output mais limpo
warnings.filterwarnings('ignore')


try:
    # ADICIONADO 'GPT_MODEL' para o tiktoken
    from .config import (
        D2L_BOOK_LINK, BOOKS_PATH, KNOWLEDGE_DIR, OCR_DPI, logger,
        NEURALMIND_API_KEY, NEURALMIND_OCR_URL, 
        NEURALMIND_API_USE, OCR_PAGES, GPT_MODEL 
    )
except ImportError:
    print("❌ Erro: Não foi possível importar 'deepmentor.config'.")
    print("   Certifique-se de que você ativou o ambiente virtual (.venv) e")
    print("   instalou o projeto com 'pip install -U -r requirements.txt' (que inclui o '-e .')")
    sys.exit(1)

app = typer.Typer(
    help="Scripts para carregar e preparar dados para o DeepMentor.",
    add_completion=False
)


@app.command()
def download_book():
    """
    Baixa o livro 'Dive into Deep Learning' (d2l-en.pdf).
    """
    
    logger.info(f"Diretório de destino: {BOOKS_PATH}")
    BOOKS_PATH.mkdir(parents=True, exist_ok=True)
    
    pdf_path = BOOKS_PATH / "d2l-en.pdf"

    if pdf_path.exists():
        logger.info(f"O arquivo '{pdf_path.name}' já existe. Download pulado.")
        return

    logger.info(f"📥 Baixando o livro de {D2L_BOOK_LINK}...")
    try:
        response = requests.get(D2L_BOOK_LINK, stream=True)
        response.raise_for_status()  # Dispara erro se o download falhar
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"✅ Download concluído! Arquivo salvo em: {pdf_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Falha no download: {e}")
        raise typer.Exit(code=1)


@app.command()
def prepare_images():
    """
    Converte as páginas do PDF 'd2l-en.pdf' em imagens PNG.
    """
    logger.info("Iniciando conversão de PDF para PNG...")
    
    # Caminhos
    pdf_path = BOOKS_PATH / "d2l-en.pdf"
    output_dir = BOOKS_PATH / "pages"
    ocrdata_file = KNOWLEDGE_DIR / "d2l-ocr.json"

    # Garante que os diretórios existam
    output_dir.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        logger.error(f"❌ PDF não encontrado em {pdf_path}.")
        logger.error("Rode 'python3 -m deepmentor.data_loader download-book' primeiro.")
        raise typer.Exit(code=1)

    try:
        total_pages = len(PdfReader(pdf_path).pages)
    except Exception as e:
        logger.error(f"❌ Erro ao ler o PDF: {e}")
        raise typer.Exit(code=1)

    # Carregar resultados prévios (se houver)
    if ocrdata_file.exists():
        with open(ocrdata_file, "r", encoding="utf-8") as f:
            summary_dict = json.load(f)
    else:
        summary_dict = {}

    if not summary_dict:
        logger.info(f"📘 Processando {total_pages} páginas...")

    pages_converted = 0
    for i in tqdm(range(1, total_pages + 1), desc="Convertendo PDF"):
        page_key = str(i)
        if page_key in summary_dict:
            continue  # Página já processada e registrada no JSON

        img_path = output_dir / f"page_{i:04d}.png"

        # Converter para PNG se ainda não existir
        if not img_path.exists():
            try:
                images = convert_from_path(pdf_path, dpi=OCR_DPI, first_page=i, last_page=i)
                if not images:
                    logger.warning(f"⚠️ Falha ao converter página {i}")
                    continue
                images[0].save(img_path, "PNG")
                pages_converted += 1
            except Exception as e:
                logger.error(f"❌ Erro ao converter página {i}: {e}")
                continue

    logger.info(f"✅ Conversão completa! {pages_converted} novas páginas convertidas.")
    logger.info(f"Imagens salvas em: {output_dir}")

    # AVISO: A linha abaixo (display) pode falhar em terminais puros.
    # É segura em notebooks (Jupyter, VS Code Notebooks).
    try:
        img = Image.open(output_dir / "page_0049.png")
        display(img)
    except Exception:
        logger.info("Não foi possível exibir a imagem de amostra no terminal.")


@app.command()
def run_ocr():
    """
    Executa o OCR da NeuralMind nas imagens PNG geradas.
    """
    if not NEURALMIND_API_USE:
        logger.warning("⚠️ OCR pulado. 'NEURALMIND_API_USE' está 'False' no config.py.")
        return

    logger.info("Iniciando processo de OCR com a API NeuralMind...")
    logger.info(f"Iniciando comunicação: {NEURALMIND_OCR_URL}")

    # Caminhos
    output_dir = BOOKS_PATH / "pages"
    ocrdata_file = KNOWLEDGE_DIR / "d2l-ocr.json"
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    # Carregar resultados prévios (se houver)
    if ocrdata_file.exists():
        with open(ocrdata_file, "r", encoding="utf-8") as f:
            ocr_content_dict = json.load(f)
    else:
        ocr_content_dict = {}

    # Configuração da API
    url_neural_mind = NEURALMIND_OCR_URL
    headers = {
        "accept": "application/json",
        "X-API-KEY": NEURALMIND_API_KEY
    }

    pages_processed = 0
    try:
        for page in tqdm(OCR_PAGES, desc="Seções OCR"):
            section = page["section"]
            start_page = page["start_page"]
            end = page["end"]

            # Garante que a seção exista no dicionário
            if section not in ocr_content_dict:
                ocr_content_dict[section] = {}

            for i in tqdm(range(start_page, end + 1), desc=f"OCR ({section})", leave=False):
                if str(i) in ocr_content_dict[section]:
                    continue

                page_key = f"page_{i:04d}.png"
                img_path = output_dir / page_key

                if not img_path.exists():
                    logger.warning(f"⚠️ Imagem {img_path} não encontrada. Pule 'prepare-images'?")
                    continue

                files = {"image": (page_key, open(img_path, "rb"), "image/png")}
                data = {
                    "prompt": f"<image>\nFree OCR.",
                    "temperature": "0", "max_tokens": "8192", "ngram_size": "30",
                    "window_size": "90", "skip_special_tokens": "false"
                }

                try:
                    response = requests.post(
                        url_neural_mind, headers=headers, files=files, data=data, timeout=300
                    )
                    response.raise_for_status()
                    result = response.json()
                    ocr_content_dict.setdefault(section, {})[str(i)] = {"text": result['text']}
                    pages_processed += 1
                except requests.exceptions.RequestException as e:
                    logger.error(f"❌ Erro no OCR para página {i}: {e}")
    finally:
        # Salva os resultados
        if pages_processed > 0:
            with open(ocrdata_file, "w", encoding="utf-8") as f:
                json.dump(ocr_content_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ OCR Concluído. {pages_processed} novas páginas processadas.")
            logger.info(f"Resultados salvos em {ocrdata_file}")
        else:
            logger.info("✅ OCR Concluído. Nenhuma página nova processada.")


try:
    _encoder = tiktoken.encoding_for_model(GPT_MODEL)
except Exception:
    logger.warning(f"Falha ao carregar encoding do GPT_MODEL '{GPT_MODEL}'. Usando 'cl100k_base'.")
    _encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(string: str) -> int:
    """Retorna o número de tokens em uma string de texto."""
    if not string:
        return 0
    return len(_encoder.encode(string))

@app.command()
def calculate_tokens():
    """
    Calcula os tokens (usando tiktoken) para o 'd2l-ocr.json'.
    
    Lê o arquivo JSON gerado pelo 'run-ocr', conta os tokens
    para cada página e exibe um relatório por seção.
    """
    logger.info(f"Iniciando contagem de tokens (usando '{_encoder.name}')...")

    # Caminho do arquivo de dados
    ocrdata_file = KNOWLEDGE_DIR / "d2l-ocr.json"

    if not ocrdata_file.exists():
        logger.error(f"❌ Arquivo OCR não encontrado em {ocrdata_file}")
        logger.error("   Execute 'python3 -m deepmentor.data_loader run-ocr' primeiro.")
        raise typer.Exit(code=1)

    # Carregar o arquivo JSON
    try:
        with open(ocrdata_file, "r", encoding="utf-8") as f:
            ocr_content_dict = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Erro ao ler o JSON: {e}")
        raise typer.Exit(code=1)
    
    if not ocr_content_dict:
        logger.warning("⚠️ O arquivo JSON está vazio. Nada para contar.")
        return

    logger.info("--- Relatório de Tokens ---")
    token_report = {}
    total_geral = 0

    for section, pages in ocr_content_dict.items():
        if not isinstance(pages, dict):
            logger.warning(f"Seção '{section}' mal formatada. Pulando.")
            continue
            
        token_report[section] = {}
        section_total = 0
        
        logger.info(f"\nCalculando Seção: '{section}'")
        for page, content in pages.items():
            text = content.get("text", "")
            num_tokens = count_tokens(text)
            token_report[section][page] = num_tokens
            section_total += num_tokens
            logger.info(f"  Página: {page} → {num_tokens} tokens")

        # Total por seção
        logger.info(f"  Total da Seção '{section}': {section_total} tokens")
        total_geral += section_total
    
    logger.info("\n--- Resumo Total ---")
    logger.info(f"  Total Geral (todas as seções): {total_geral} tokens")
    logger.info("✅ Contagem de tokens concluída.")

if __name__ == "__main__":
    app()