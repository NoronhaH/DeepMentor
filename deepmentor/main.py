import os
import sys
import asyncio
from pathlib import Path
import shutil
from shutil import copy2

from deepmentor.config import KNOWLEDGE_DIR

# Adiciona a Raiz do Projeto ao Path
# Essencial para encontrar o módulo 'flows' e 'deepmentor'
PROJ_ROOT = Path(__file__).resolve().parents[1] # Sobe 1 nível
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    # Importa o logger do config
    from deepmentor.config import PROJ_ROOT, KNOWLEDGE_DIR_DYN, logger
    
    # Importa a classe do flow (CrewAI)
    from flows.deepmentor_flow.crews.teaching_crew.teaching_crew import DeepMentorFlow

except ImportError as e:
    print(f"❌ Erro: Não foi possível importar os módulos. {e}")
    print("   Certifique-se de que o .venv está ativo e as dependências estão instaladas.")
    sys.exit(1)


#async def run_flow():
def run_flow():
    """
    Instancia, plota e executa o DeepMentorFlow.
    """

    # Prepara o knowledge (dinâmico)
    KNOWLEDGE_DIR_DYN.mkdir(parents=True, exist_ok=True) # Cria diretório se não existe

    # Efetua cópia do knowledge estático para o knowledge dinâmico
    shutil.copy2(KNOWLEDGE_DIR / "d2l-ocr.json", KNOWLEDGE_DIR_DYN / "d2l-ocr.json")

    # Instanciação do Flow
    deep_mentor_flow = DeepMentorFlow()
    
    # Lógica do Plot (salvando em docs/)
    docs_dir = PROJ_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir()

    logger.info(f"Gerando gráfico de fluxo...")

    # Retorna o caminho do HTML temporário
    temp_html_path = Path(deep_mentor_flow.plot(show=False))
    temp_dir = temp_html_path.parent

    logger.info(f"Plot temporário criado em: {temp_dir}")

    # Copia arquivos
    for filename in [
        "crewai_flow.html",
        "crewai_flow_script.js",
        "crewai_flow_style.css"
    ]:
        src = temp_dir / filename
        dst = docs_dir / filename

        if src.exists():
            copy2(src, dst)

    # Renomear o HTML principal
    final_html = docs_dir / "DeepMentorFlow.html"
    os.rename(docs_dir / "crewai_flow.html", final_html)

    logger.info(f"Plot final salvo em: {final_html}")

    # Inicialização do Flow
    # - Inicializa o estado inicial do fluxo
    # - Aguarda primeira interação do usuário para iniciar o fluxo
    deep_mentor_flow.initialize()

    print("\n")

if __name__ == "__main__":
    logger.info("Iniciando executor principal (deepmentor/main.py)...")
    #asyncio.run(run_flow())
    run_flow()