# Executar:
# python3 -m deepmentor.tester --test_self_hosted
# python3 -m deepmentor.tester --test-agents

import os
import shutil
import typer
import time
import sys
import warnings
import sqlite3
from pathlib import Path

# Ignora warnings para um output mais limpo
warnings.filterwarnings('ignore')

try:
    from .config import (
        OLLAMA_API_BASE, KNOWLEDGE_DIR, LONG_TERM_MEMORY_DIR, 
        GPT_MODEL, OPENAI_API_KEY, OLLAMA_MODEL, logger
    )
except ImportError:
    print("❌ Erro: Não foi possível importar 'deepmentor.config'.")
    print("   Certifique-se de que você ativou o ambiente virtual (.venv) e")
    print("   instalou o projeto com 'pip install -U -r requirements.txt' (que inclui o '-e .')")
    sys.exit(1)

try:
    from langchain_community.chat_models import ChatOllama
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
    from crewai.memory import LongTermMemory
    from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
except ImportError:
    print("❌ Erro: Bibliotecas 'langchain_community' ou 'crewai' não encontradas.")
    print("   Certifique-se de que as dependências estão instaladas.")
    sys.exit(1)


# Teste 1: Conexão Ollama
def test_ollama_langchain(model_name: str, base_url: str):
    """Testa a conexão com um modelo Ollama via LangChain."""
    
    print(f"🔍 Testando ChatOllama com {model_name}...")
    
    if not base_url:
        print(f"❌ Erro: 'OLLAMA_API_BASE' (base_url) não foi configurado no seu .env")
        return False
        
    try:
        llm = ChatOllama(
            model=model_name,
            base_url=base_url
        )
        start_time = time.time()
        
        response = llm.invoke("ping") 
        latency = time.time() - start_time
        
        print(f"✅ Modelo {model_name} respondeu em {latency:.2f}s")
        print(f"   Resposta: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao conectar ao modelo {model_name}: {e}")
        return False


# Teste 2: Agentes e LTM
def query_ltm_database(db_path: Path):
    """Consulta e imprime o conteúdo da tabela LTM do SQLite."""
    
    print("\n--- Consultando o Banco de Dados LTM ---")
    if not db_path.exists():
        print(f"❌ Erro: Arquivo de banco de dados não encontrado em {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print(f"Conectado ao banco de dados SQLite: {db_path}")

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            print("Nenhuma tabela encontrada no banco de dados.")
            return

        print(f"Tabelas encontradas: {tables}")

        # Consulta a tabela principal do LTM
        cursor.execute("SELECT * FROM long_term_memories")
        
        column_names = [description[0] for description in cursor.description]
        print("\nNomes das Colunas (long_term_memories):", column_names)

        rows = cursor.fetchall()
        if rows:
            print("\nConteúdo da tabela long_term_memories:")
            for row in rows:
                # Imprime apenas os primeiros 100 caracteres de dados para não poluir
                print(f"  ID: {row[0]}, Dados: {str(row[1])[:100]}...")
        else:
            print("\nNenhum dado encontrado na tabela long_term_memories.")

    except sqlite3.Error as e:
        print(f"Erro ao conectar ou consultar o banco de dados SQLite: {e}")
    finally:
        if conn:
            conn.close()
            print("\nConexão SQLite fechada.")


def test_agent_knowledge():
    """Executa um teste completo do CrewAI com Agente, Knowledge e LTM."""
    
    logger.info("Iniciando teste de Agente, Knowledge e LTM do CrewAI...")

    # Definir Caminhos
    json_file_path = KNOWLEDGE_DIR / "d2l-ocr.json"
    db_path = LONG_TERM_MEMORY_DIR / "test_memory.db"
    
    os.makedirs("knowledge", exist_ok=True)
    shutil.copyfile(
        src=json_file_path,
        dst="knowledge/d2l-ocr.json"
    )

    # Verifica se o arquivo de knowledge existe
    if not json_file_path.exists():
        logger.error(f"❌ Arquivo de knowledge não encontrado: {json_file_path}")
        logger.error("   Execute 'python3 -m deepmentor.data_loader run-ocr' primeiro.")
        return False

    # Configurar Knowledge
    logger.info(f"Carregando knowledge de {json_file_path}...")
    d2l_book_knowledge = JSONKnowledgeSource(
        file_paths=["d2l-ocr.json"]
    )

    # Configurar Agente
    logger.info("Configurando Agente de Teste...")
    testAgent = Agent(
        role='Test Agent',
        goal="Help the user provide information for a simple test.",
        backstory="You are merely a simple test agent. Help the user provide information for a test.",
        knowledge_sources=[d2l_book_knowledge],
        llm=LLM(
            model=GPT_MODEL,
            base_url="https://api.openai.com/v1",
            api_key=OPENAI_API_KEY,
        ),
        verbose=False
    )

    # Configurar Crew
    logger.info(f"Configurando Crew (LTM será salvo em {db_path})...")
    crew_setup = Crew(
        agents=[testAgent],
        tasks=[
            Task(
                description="Help the user provide information for a test. And save things in your memory. Search for the 'summary' or table of contents in the knowledge base and provide the information from chapter 3.",
                expected_output="Provide information the requested information.",
                agent=testAgent
            ),
        ],
        memory=True,
        long_term_memory=LongTermMemory(
            storage=LTMSQLiteStorage(
                db_path=str(db_path) # Storage espera uma string
            )
        ),
        process=Process.sequential,
        verbose=True
    )

    # Executar Crew
    logger.info("🚀 Executando crew_setup.kickoff()...")
    result = crew_setup.kickoff()
    logger.info(f"🏁 Resultado do Crew: {result}")

    # Consultar LTM
    query_ltm_database(db_path)

    # Resetar Memória
    logger.info("Resetando LTM...")
    crew_setup.reset_memories(command_type='long')
    
    # Resetando diretório dinâmico "knowledge/"
    os.remove("knowledge/d2l-ocr.json")
    os.rmdir("knowledge")

    return True
 
# Criação do App Typer
app = typer.Typer(
    help="Ferramenta de testes do DeepMentor.",
    add_completion=False
)

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    test_self_hosted: bool = typer.Option(
        False, 
        "--test_self_hosted",
        help="Executa um teste de conexão com o modelo Ollama."
    ),
    test_agents: bool = typer.Option(
        False,
        "--test-agents",
        help="Executa um teste de Agente, Knowledge e LTM do CrewAI."
    )
):
    """
    Ponto de entrada principal da CLI.
    """
    if test_self_hosted:
        print("Iniciando teste de conexão self-hosted...")
                
        success = test_ollama_langchain(
            model_name=OLLAMA_MODEL,
            base_url=OLLAMA_API_BASE
        )
        
        if not success:
            print("⛔ Teste de conexão Ollama falhou.")
            raise typer.Exit(code=1)
        print("\n✅ Conexão com Ollama bem-sucedida.")
    
    elif test_agents:
        success = test_agent_knowledge()
        
        if not success:
            print("⛔ Teste de Agentes CrewAI falhou.")
            raise typer.Exit(code=1)
        print("\n✅ Teste de Agentes CrewAI concluído.")
    
    elif ctx.invoked_subcommand is None:
        print("Nenhuma ação especificada. Use --help para ver as opções.")
        print("\n   Exemplos de uso:")
        print("   python3 -m deepmentor.tester --test_self_hosted")
        print("   python3 -m deepmentor.tester --test-agents")

if __name__ == "__main__":
    app()