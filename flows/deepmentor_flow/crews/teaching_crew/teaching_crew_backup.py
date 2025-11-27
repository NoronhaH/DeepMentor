import os
import sys
import shutil
import asyncio
import yaml
from pathlib import Path
from typing import Any, Dict, List

from deepmentor.config import OLLAMA_API_BASE

# Adiciona a Raiz do Projeto ao Path
PROJ_ROOT = Path(__file__).resolve().parents[4] 
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

# Importações do Projeto e Bibliotecas
try:
    from deepmentor.config import (
        GPT_MODEL, OPENAI_API_KEY, logger,
        CREW_CONFIG_DIR, KNOWLEDGE_DIR
    )
    
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.flow.flow import Flow, listen, or_, and_, router, start

    from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
    
    from pydantic import BaseModel, Field, ValidationError

except ImportError as e:
    print(f"❌ Erro: Não foi possível importar os módulos. {e}")
    print("   Certifique-se de que o .venv está ativo e as dependências estão instaladas.")
    print("   Tente: pip install pyyaml")
    sys.exit(1)

# --------------------------------
# Definição dos Estados (Pydantic)
# --------------------------------

class OrchestratorAnalysis(BaseModel):
    turn: int = Field(description="O número do próximo turno (turno atual + 1).")
    next_agent: str = Field(description="O próximo agente a ser acionado: 'professor', 'dean', ou 'none'.")
    next_instruction: str = Field(description="A instrução clara para o próximo agente.")

class ProfessorOutput(BaseModel):
    edu_content: str = Field(description="Conteúdo de ensino gerado, se houver.")
    test_content: str = Field(description="Pergunta do teste gerada, se houver.")
    result: str = Field(description="Avaliação, score ou feedback gerado, se houver.")

class DeanOutput(BaseModel):
    teaching_plan: Dict[str, Any] = Field(description="O plano de ensino completo e atualizado.")
    teaching_plan_progress: float = Field(description="O progresso atualizado do plano (0.0 a 1.0).")
    next_instruction_for_professor: str = Field(description="A próxima instrução para o Professor, baseada nesta análise.")

class InteractionState(BaseModel):
    turn: int = 0
    agents: List[str] = []
    next_agent: str = ""
    next_instruction: str = ""
    user_message: str = ""
    edu_content: str = ""
    test_content: str = ""
    conversation_history: List[str] = []
    teaching_plan: Dict[str, Any] = {}
    teaching_plan_progress: float = 0.0
    user_feedback: str = ""
    result: str = ""

# Definição do Flow
class DeepMentorFlow(Flow[InteractionState]):
    """Define o fluxo de orquestração do DeepMentor."""

    def __init__(self):
        super().__init__()
        
        # Definir o LLM Global
        #self.llm = LLM(
        #    model=GPT_MODEL,
        #    base_url="https://api.openai.com/v1",
        #    api_key=OPENAI_API_KEY,
        #)
        self.llm = LLM(
            model="ollama/gpt-oss:20b",
            base_url=OLLAMA_API_BASE,
            temperature=0.7
        )
        logger.info(f"LLM carregado: {self.llm.model}")

        # Carregar os .yamls
        config_path = str(CREW_CONFIG_DIR)
        agent_file = os.path.join(config_path, 'agents.yaml')
        task_file = os.path.join(config_path, 'tasks.yaml')

        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                self.agent_definitions = yaml.safe_load(f)
            with open(task_file, 'r', encoding='utf-8') as f:
                self.task_definitions = yaml.safe_load(f)
            logger.info("Arquivos agents.yaml e tasks.yaml carregados.")

        except Exception as e:
            logger.error(f"Erro ao carregar arquivos .yaml: {e}")
            sys.exit(1)

        # Carrega a fonte de Knowledge
        json_file_path = KNOWLEDGE_DIR / "d2l-ocr.json"
        if not json_file_path.exists():
            logger.error(f"Arquivo de knowledge não encontrado: {json_file_path}")
            logger.error("Execute 'python3 -m deepmentor.data_loader run-ocr' primeiro.")
            sys.exit(1)
        
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

        self.d2l_book_knowledge = JSONKnowledgeSource(
            file_paths=["d2l-ocr.json"]
        )
        logger.info("Fonte de Knowledge (d2l-ocr.json) carregada.")


    # Fluxo de Execução (Start)
    @start()
    def start_flow(self):
        """Etapa inicial que define o estado."""
        logger.info("🚀 DeepMentor v1.0 iniciado.")
        
        self.state.turn = 0
        self.state.next_agent = "orchestrator"
        self.state.next_instruction = "ordering" # Instrução inicial
        self.state.teaching_plan_progress = 0.0
        
        logger.info("Estado inicial definido. Próximo agente: Orquestrador")
        return "orchestrator"

    @router(start_flow)
    async def orchestrator_router(self) -> str:
        """
        O nó do Orquestrador. Analisa o estado e decide o próximo agente.
        """
        logger.info(f"🎯 Turno {self.state.turn}: Agente Orquestrador")

        # Instanciação Dinâmica
        agent_def = self.agent_definitions['orchestrator']
        task_def = self.task_definitions['orchestrator_task']

        orchestrator_agent = Agent(
            role=agent_def['role'],
            goal=agent_def['goal'],
            backstory=agent_def['backstory'],
            verbose=agent_def.get('verbose', False),
            llm=self.llm
            # O Orquestrador não precisa de knowledge
        )

        task_template = Task(
            description=task_def['description'].format(
                turn=self.state.turn,
                next_agent=self.state.next_agent,
                next_instruction=self.state.next_instruction,
                user_message=self.state.user_message,
                teaching_plan_progress=self.state.teaching_plan_progress,
                teaching_plan=self.state.teaching_plan,
                conversation_history="\n".join(self.state.conversation_history)
            ),
            expected_output=task_def['expected_output'],
            agent=orchestrator_agent,
            output_pydantic=OrchestratorAnalysis  
        )
        
        crew = Crew(
            agents=[orchestrator_agent],
            tasks=[task_template],
            process=Process.sequential,
            verbose=True
        )
        result = await crew.kickoff_async()
        
        try:
            analysis = result.pydantic
            if not analysis:
                raise ValueError("O resultado Pydantic do Orquestrador foi None.")

            self.state.turn = analysis.turn
            self.state.next_agent = analysis.next_agent
            self.state.next_instruction = analysis.next_instruction

        except (ValidationError, TypeError, ValueError) as e:
            logger.error(f"❌ O Orquestrador retornou dados inválidos: {e}")
            logger.error(f"   Resposta bruta: {result.raw}")
            return "none" 
         
        logger.info(f"Orquestrador decidiu: Próximo Agente: {self.state.next_agent}, Instrução: {self.state.next_instruction}")
        
        return self.state.next_agent.lower() 


    @listen(orchestrator_router)
    async def dean_node(self) -> str:
        """Executa o Agente Dean."""
        logger.info(f"🏛️ Turno {self.state.turn}: Agente Dean (Diretor)")

        agent_def = self.agent_definitions['dean']
        task_def = self.task_definitions['dean_task']

        dean_agent = Agent(
            role=agent_def['role'],
            goal=agent_def['goal'],
            backstory=agent_def['backstory'],
            knowledge_sources=[self.d2l_book_knowledge], # <--- KNOWLEDGE ADICIONADO
            llm=self.llm
        )
        
        task_template = Task(
            description=task_def['description'].format(
                instruction=self.state.next_instruction,
                teaching_plan=self.state.teaching_plan,
                teaching_plan_progress=self.state.teaching_plan_progress,
                result=self.state.result 
            ),
            expected_output=task_def['expected_output'],
            agent=dean_agent,
            output_pydantic=DeanOutput
        )

        crew = Crew(agents=[dean_agent], tasks=[task_template], process=Process.sequential)
        result = await crew.kickoff_async()

        try:
            analysis = result.pydantic
            if not analysis:
                raise ValueError("O resultado Pydantic do Dean foi None.")

            self.state.teaching_plan = analysis.teaching_plan
            self.state.teaching_plan_progress = analysis.teaching_plan_progress
            self.state.next_instruction = analysis.next_instruction_for_professor

            self.state.conversation_history.append(f"Dean: Plano atualizado, progresso {self.state.teaching_plan_progress}")

        except (ValidationError, TypeError, ValueError) as e:
            logger.error(f"❌ O Dean retornou dados inválidos: {e}")
            logger.error(f"   Resposta bruta: {result.raw}")
            self.state.next_instruction = "Erro do Dean: Tente novamente."

        # Dean SEMPRE retorna ao Orquestrador para que ele saiba o novo plano
        return "orchestrator" 


    @listen(or_(orchestrator_router, dean_node))
    async def professor_node(self) -> str:
        """Executa o Agente Professor."""
        logger.info(f"👨‍🏫 Turno {self.state.turn}: Agente Professor")
        
        agent_def = self.agent_definitions['professor']
        task_def = self.task_definitions['professor_task']
        
        professor_agent = Agent(
            role=agent_def['role'],
            goal=agent_def['goal'],
            backstory=agent_def['backstory'],
            knowledge_sources=[self.d2l_book_knowledge],
            llm=self.llm
        )
        
        task_template = Task(
            description=task_def['description'].format(
                instruction=self.state.next_instruction,
                teaching_plan=self.state.teaching_plan,
                teaching_plan_progress=self.state.teaching_plan_progress,
                conversation_history="\n".join(self.state.conversation_history),
                user_message=self.state.user_message
            ),
            expected_output=task_def['expected_output'],
            agent=professor_agent,
            output_pydantic=ProfessorOutput
        )

        crew = Crew(agents=[professor_agent], tasks=[task_template], process=Process.sequential)
        result = await crew.kickoff_async()
        
        # Salva o resultado do Professor no estado
        # O 'result.pydantic' é um objeto ProfessorOutput
        if result.pydantic:
            self.state.edu_content = result.pydantic.edu_content
            self.state.test_content = result.pydantic.test_content
            self.state.result = result.pydantic.result # Para o Dean
        
        self.state.conversation_history.append(f"Professor: {result.pydantic}")

        # MUDANÇA: Em vez de voltar ao orquestrador, envia para a interface do usuário
        return "user_interface" 


    @listen(professor_node)
    async def user_interface_node(self) -> str:
        """
        Simula a interface do chat. Exibe a saída do Professor
        e (por enquanto) simula uma resposta do usuário para
        continuar o fluxo.
        """
        logger.info("🗣️  Interface do Usuário (Chat)")

        # Exibe o que o Professor disse
        if self.state.edu_content:
            print(f"\n[PROFESSOR DIZ]:\n{self.state.edu_content}\n")
            self.state.edu_content = "" # Limpa o estado
        
        if self.state.test_content:
            print(f"\n[PROFESSOR APLICA TESTE]:\n{self.state.test_content}\n")
            self.state.test_content = "" # Limpa o estado
        
        if self.state.result and not (self.state.edu_content or self.state.test_content):
            # Se 'result' for a única coisa (ex: feedback de um teste)
            print(f"\n[PROFESSOR AVALIA]:\n{self.state.result}\n")
            self.state.result = "" # Limpa o estado


        logger.info("... (Simulando resposta do usuário após 2 segundos)...")
        await asyncio.sleep(2)
        
        simulated_response = "Ok, entendi. Pode me passar o próximo tópico?"
        self.state.user_message = simulated_response
        self.state.conversation_history.append(f"User: {simulated_response}")
        logger.info(f"User (Simulado): {simulated_response}")

        # Envia de volta ao Orquestrador para analisar a resposta do usuário
        return "orchestrator"

    @listen(orchestrator_router)
    def final_step(self):
        """Nó final (quando o Orquestador decide encerrar)."""
        logger.info("🏁 Fluxo chegou ao fim (end_session).")