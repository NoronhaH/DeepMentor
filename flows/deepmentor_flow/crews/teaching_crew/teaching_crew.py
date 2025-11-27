import os
from signal import pause
import sys
import time
import shutil
import json
import re
import asyncio
import threading
import yaml
import tiktoken
from pathlib import Path
from typing import Any, Dict, List, Tuple
from textwrap import dedent

# Adiciona a Raiz do Projeto ao Path
PROJ_ROOT = Path(__file__).resolve().parents[4] 
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

# Importações do Projeto e Bibliotecas
try:
    from deepmentor.config import (
        GPT_MODEL, OPENAI_API_KEY, logger,
        CREW_CONFIG_DIR, KNOWLEDGE_DIR, OLLAMA_API_BASE, LLM_MODEL, OLLAMA_MODEL,
        GRADIO_PUBLIC_SHARE
    )
    
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.flow.flow import Flow, listen, or_, and_, router, start
    #from crewai.project import CrewBase, agent, task, crew, llm, before_kickoff, after_kickoff
    from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
    from pydantic import BaseModel, Field, ValidationError, ConfigDict

    import gradio as gr

except ImportError as e:
    print(f"❌ Erro: Não foi possível importar os módulos. {e}")
    print("   Certifique-se de que o .venv está ativo e as dependências estão instaladas.")
    print("   Tente: pip install pyyaml")
    sys.exit(1)

# --------------------------------
# Definição dos Estados (Pydantic)
# --------------------------------
class GeneratorExamplesOutput(BaseModel):
    examples: List[List[str]] = Field(description="Exemplos de conversação com base no tema/tópico/histórico para o usuário.")

class PresentationOutput(BaseModel):
    presentation_message: str = Field(description="Mensagem completa de apresentação para o aluno (em markdown).")
    user_name: str = Field(default="", description="Nome do usuário informado, ou vazio se ainda não foi informado.")
    available_topics: List[str] = Field(default_factory=list, description="Lista de tópicos disponíveis na base de conhecimento.")
    user_interest_topics: List[str] = Field(default_factory=list, description="Tópicos de interesse do usuário.")
    user_focus_type: str = Field(default="", description="Tipo de foco desejado: 'teórico', 'prático', 'equilibrado' ou vazio.")
    user_level: str = Field(default="", description="Nível do usuário: 'iniciante', 'intermediário', 'avançado' ou vazio.")
    
    # Campos de satisfação para cada informação necessária
    user_name_satisfied: str = Field(description="Status da coleta do nome: 'satisfied' ou 'not_satisfied'.")
    user_interest_satisfied: str = Field(description="Status da coleta de interesses: 'satisfied' ou 'not_satisfied'.")
    topic_selection_satisfied: str = Field(description="Status da seleção de tema: 'satisfied' ou 'not_satisfied'.")
    user_focus_satisfied: str = Field(description="Status da coleta do tipo de foco: 'satisfied' ou 'not_satisfied'.")
    user_level_satisfied: str = Field(description="Status da coleta do nível: 'satisfied' ou 'not_satisfied'.")
    
    # Status geral
    all_requirements_met: bool = Field(default=False, description="True se todas as informações necessárias foram coletadas.")
    next_suggested_action: str = Field(description="Próxima ação sugerida.")

class OrchestratorAnalysis(BaseModel):
    next_instruction: str = Field(description="A instrução/ação a executar: 'subject_choice', 'teaching_plan_ordering', 'end_session', etc.")

class ProfessorOutput(BaseModel):
    message_presentation: str = Field(description="Mensagem principal de apresentação para o usuário (sempre preenchido).")
    edu_content: str = Field(default="", description="Conteúdo de ensino gerado, se houver.")
    test_content: str = Field(default="", description="Descrição/pergunta do teste gerada, se houver.")
    test_code: str = Field(default="", description="Código inicial/template para o teste, se houver.")
    result: str = Field(default="", description="Avaliação, score ou feedback gerado, se houver.")
    mode: str = Field(description="Modo atual: 'teaching' (ensinando), 'testing' (aplicando teste), 'evaluating' (avaliando)")

class DeanOutput(BaseModel):
    teaching_plan: Dict[str, bool] = Field(description="O plano de ensino completo e atualizado. Cada tópico é mapeado para um booleano indicando se foi concluído (True) ou não (False).")
    teaching_plan_progress: float = Field(description="O progresso atualizado do plano (0.0 a 1.0).")
    next_instruction_for_professor: str = Field(description="A próxima instrução para o Professor, baseada nesta análise.")

class TeachingPlanConfirmationOutput(BaseModel):
    confirmation_message: str = Field(description="Mensagem apresentando o plano de ensino e solicitando confirmação do aluno.")
    plan_approved: bool = Field(description="Se o aluno aprovou o plano de ensino (true) ou solicitou revisão (false).")
    revision_feedback: str = Field(default="", description="Feedback do aluno sobre o que deve ser ajustado no plano, se houver.")

class InteractionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    turn: int = 0
    agents: List[str] = []

    last_agent: str = ""
    last_instruction: str = ""
    next_agent: str = ""
    next_instruction: str = ""

    user_name: str = ""
    user_interest_topics: List[str] = []
    user_focus_type: str = ""  # Tipo de foco: "teórico", "prático", "equilibrado"
    user_level: str = ""  # Nível: "iniciante", "intermediário", "avançado"
    available_topics: List[str] = []
    user_message: str = ""
    agent_response: str = ""
    edu_content: str = ""
    test_content: str = ""
    test_code: str = ""  # Código do teste atual
    user_code: str = ""  # Código submetido pelo usuário
    conversation_history: List[str] = []
    
    teaching_plan: Dict[str, bool] = {}  # {tópico: concluído (True/False)}
    teaching_plan_progress: float = 0.0
    
    user_feedback: str = ""
    result: str = ""

    # Base de Conhecimento (conteúdo carregado)
    summary_d2l: str = ""
    chapter_3_d2l: str = ""  # Mudado de List[str] para str pois é um objeto/string

    # Interface do Gradio (Variável Global)
    interface: gr.Interface = None
    chatbot: gr.Chatbot = None
    msg_input: gr.Textbox = None
    send_btn: gr.Button = None
    clear_btn: gr.Button = None
    code_output: gr.Code = None
    teaching_plan_checklist: gr.CheckboxGroup = None
    progress_bar: gr.Slider = None
    examples_dataset: gr.Dataset = None
    
    # Exemplos atuais (podem ser atualizados dinamicamente)
    current_examples: List[List[str]] = []

# --------------------------------
# Função de Contagem de Tokens
# --------------------------------
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Conta o número de tokens em um texto usando tiktoken.
    
    Args:
        text: Texto para contar tokens
        model: Modelo do GPT para usar o encoding correto
    
    Returns:
        Número de tokens
    """
    if not text:
        return 0
    
    try:
        encoder = tiktoken.encoding_for_model(model)
    except Exception:
        logger.warning(f"Falha ao carregar encoding do modelo '{model}'. Usando 'cl100k_base'.")
        encoder = tiktoken.get_encoding("cl100k_base")
    
    return len(encoder.encode(text))

# Definição do Flow
class DeepMentorFlow(Flow[InteractionState]):
    """Define o fluxo de orquestração do DeepMentor."""

    # Construtor da Classe DeepMentorFlow
    # - Carrega a base de conhecimento
    # - Carrega o LLM
    # - Carrega os arquivos de configuração (.yaml)
    # - Define exemplos padrão (fallback)
    def __init__(self):
        super().__init__()

        # Carrega o conteúdo do livro D2L para usar como contexto nas tasks
        d2l_json_path = KNOWLEDGE_DIR / "d2l-ocr.json"
        logger.info(f"📚 Carregando conteúdo do D2L de: {d2l_json_path}")
        
        with open(d2l_json_path, 'r', encoding='utf-8') as f:
            self.d2l_data = json.load(f)  # Armazena o objeto completo para uso posterior
            self.state.summary_d2l = json.dumps(self.d2l_data.get('summary', {}), ensure_ascii=False, indent=2)
            self.state.chapter_3_d2l = json.dumps(self.d2l_data.get('chapter-3', {}), ensure_ascii=False, indent=2)
        
        # Contagem de tokens e caracteres
        summary_chars = len(self.state.summary_d2l)
        summary_tokens = count_tokens(self.state.summary_d2l, model=GPT_MODEL)
        chapter3_chars = len(self.state.chapter_3_d2l)
        chapter3_tokens = count_tokens(self.state.chapter_3_d2l, model=GPT_MODEL)
        
        logger.info(f"✅ Summary carregado: {summary_chars:,} caracteres | {summary_tokens:,} tokens")
        logger.info(f"✅ Chapter 3 carregado: {chapter3_chars:,} caracteres | {chapter3_tokens:,} tokens")
        logger.info(f"📊 Total: {summary_chars + chapter3_chars:,} caracteres | {summary_tokens + chapter3_tokens:,} tokens")

        # Carrega o LLM
        if LLM_MODEL == 0:
            self.llm = LLM(
                model=GPT_MODEL,
                base_url="https://api.openai.com/v1",
                api_key=OPENAI_API_KEY,
            )
        else:
            self.llm = LLM(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_API_BASE,
                temperature=0.7
            )

        # Carrega os arquivos de configuração (.yaml)
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

        # Define exemplos padrão (fallback)
        self.default_examples = [
            ["Olá, como você está?"],
            ["Gostaria de aprender sobre Deep Learning!"],
            ["Gostaria de aprender sobre redes neurais profundas!"],
            ["Gostaria de aprender sobre otimização de redes neurais!"]
        ]
        
        # Inicializa o estado com os exemplos padrão
        self.state.current_examples = self.default_examples

    # Inicializa o estado inicial do fluxo
    def initialize(self):
        """Inicializa o estado inicial do fluxo."""
        # Instancia a Interface Gradio
        # - Cria a Interface Gradio
        # - Aguarda primeira interação do usuário para iniciar o fluxo (gerenciada pelo chat_callback)
        self.instance_gradio_interface()

    # Instancia a Interface Gradio
    def instance_gradio_interface(self):

        # Instancia o Gradio Interface
        print("🌐 Criando Gradio Interface...")
        with gr.Blocks(theme=gr.themes.Soft()) as self.state.interface:
            gr.Markdown(
                "<center><h1>"
                "🎓 DeepMentor: "
                "Ensino Adaptativo de Deep Learning"
                "</h1></center>"
            )
            
            with gr.Row():
                with gr.Column(scale=2):                    
                    # Componentes do chat
                    self.state.chatbot = gr.Chatbot(
                        value=[],
                        height=500,
                        type="messages",
                        label="Conversa"
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=9):
                            self.state.msg_input = gr.Textbox(
                                placeholder="Digite sua mensagem aqui...",
                                #label="Digite sua mensagem aqui...",
                                show_label=False,
                                submit_btn=False,
                                autofocus=True,
                                lines=2,
                                scale=4
                            )

                            # Dataset de exemplos (atualizável dinamicamente)
                            gr.Markdown("**💡 Sugestões:**")
                            self.state.examples_dataset = gr.Dataset(
                                components=[self.state.msg_input],
                                samples=self.state.current_examples,
                                label="Clique para usar",
                                type="index"
                            )
                            
                            # Botão para atualizar exemplos (teste)
                            #update_examples_btn = gr.Button("🔄 Gerar Novos Exemplos", size="sm")
                    
                        with gr.Column(scale=1):
                            self.state.send_btn = gr.Button("📤 Enviar", variant="primary", scale=1)
                            #self.state.clear_btn = gr.Button("🗑️ Limpar")
                
                with gr.Column(scale=1):
                    self.state.code_output = gr.Code(
                        label="Código Gerado",
                        language="python",
                        interactive=True,
                        lines=10
                    )

                    # Checklist para mostrar o plano de ensino (bloco dinâmico: atualizado a cada iteração)
                    gr.Markdown("**📚 Plano de Ensino:**")
                    self.state.teaching_plan_checklist = gr.CheckboxGroup(
                        choices=[],
                        value=[],  # Itens marcados (será atualizado conforme o progresso)
                        label="Tópicos do Plano",
                        interactive=False  # Apenas para visualização
                    )
                    
                    # Progress bar para visualizar o progresso geral
                    self.state.progress_bar = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        label="Progresso (%)",
                        interactive=False
                    )
            
            # Eventos
            self.state.send_btn.click(
                self.chat_callback, 
                [self.state.msg_input, self.state.chatbot, self.state.code_output], 
                [
                    self.state.msg_input,
                    self.state.chatbot,
                    self.state.teaching_plan_checklist,
                    self.state.progress_bar,
                    self.state.examples_dataset,
                    self.state.code_output
                ]
            )

            self.state.msg_input.submit(
                self.chat_callback, 
                [self.state.msg_input, self.state.chatbot, self.state.code_output], 
                [
                    self.state.msg_input,
                    self.state.chatbot,
                    self.state.teaching_plan_checklist,
                    self.state.progress_bar,
                    self.state.examples_dataset,
                    self.state.code_output
                ]
            )

            #self.state.clear_btn.click(
            #    lambda: ([], ""),
            #    None,
            #    [self.state.chatbot, self.state.msg_input]
            #)
            
            # Evento para clicar em exemplo
            self.state.examples_dataset.select(
                self.select_example,
                None,  # Não precisa de inputs
                [self.state.msg_input]
            )
            
            # Evento para atualizar exemplos
            """update_examples_btn.click(
                self.update_examples,
                None,
                [examples_dataset]
            )"""
        
        print("✅ Gradio Interface criada!")

        self.state.interface.launch(
            prevent_thread_lock=False,
            share=GRADIO_PUBLIC_SHARE
        )
        
        logger.info("🚀 DeepMentor v1.0 iniciado.")


    # Callback do chat
    def chat_callback(self, message: str, history: list, user_code: str = ""):
        """Processa mensagens do chat e retorna todos os componentes atualizados."""
        logger.info(f"💬 Mensagem recebida: {message}")
        
        # Captura o código do usuário se houver
        if user_code and user_code.strip():
            self.state.user_code = user_code
            logger.info(f"💻 Código capturado do usuário ({len(user_code)} caracteres)")
        
        # Atualiza o estado com a mensagem do usuário ANTES de processar
        self.state.user_message = message
        self.state.conversation_history.append(f"User: {message}")
        
        # Se turn == 0, faz o kickoff do fluxo inicial
        if self.state.turn == 0:
            self.kickoff()  # Inicia o fluxo (executa start_flow e deepmentor_presentation)
            # Após o kickoff, o estado já foi atualizado com a resposta

        if self.state.turn > 0 and self.state.next_instruction == "teaching_plan_ordering":
            # Inicia o flow orchestrator
            self.kickoff()
        # Se turn > 0 e ainda está na fase de apresentação, continua coletando informações
        elif self.state.next_instruction == "user message":
            logger.info("🔄 Continuando coleta de informações do usuário...")
            # Executa novamente a crew de apresentação para processar a nova mensagem
            self._execute_presentation_crew()
        # Se está aguardando confirmação do plano de ensino
        elif self.state.next_instruction == "teaching_plan_confirmation":
            logger.info("📋 Processando confirmação do plano de ensino...")
            # Executa a crew de confirmação
            self._execute_teaching_plan_confirmation_crew()
            # Se foi aprovado ou precisa de revisão, continua o fluxo
            if self.state.next_instruction == "start_teaching":
                logger.info("✅ Plano aprovado, iniciando ensino")
                # Continua o fluxo normal
                self.kickoff()
            elif self.state.next_instruction == "teaching_plan_revision":
                logger.info("🔄 Revisando plano conforme feedback")
                # Volta para o Dean revisar
                self.kickoff()
        # Se está em modo de ensino contínuo (Professor aguardando resposta)
        elif self.state.next_instruction == "user_message" and self.state.last_instruction == "continue_teaching":
            logger.info("📚 Continuando ensino com Professor...")
            # Chama o professor diretamente para processar a mensagem do usuário
            # Atualiza a instrução para indicar que está no modo teaching
            self.state.next_instruction = "teaching"
            self.kickoff()  # Vai direto para o professor via router
        
        # Retorna a resposta do agente atual
        response = self.state.agent_response
        
        # Incrementa o turno
        self.state.turn += 1
        
        # Adiciona resposta ao histórico interno
        self.state.conversation_history.append(f"Assistant ({self.state.last_agent}): {response}")
        
        # Atualiza o histórico do chat
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # Atualiza o checklist baseado no progresso
        # Extrai os tópicos do teaching_plan se existir, senão usa lista padrão
        if self.state.teaching_plan and isinstance(self.state.teaching_plan, dict):
            topics_list = list(self.state.teaching_plan.keys())
            # Identifica os tópicos concluídos baseado nos valores booleanos
            completed_indices = [i for i, topic in enumerate(topics_list) if self.state.teaching_plan[topic]]
            
            # Log para debug
            logger.debug(f"📋 Checklist: {len(topics_list)} tópicos, {len(completed_indices)} concluídos")
            for i, (topic, completed) in enumerate(self.state.teaching_plan.items()):
                logger.debug(f"   {i+1}. [{('✅' if completed else '⬜')}] {topic}")
        else:
            topics_list = []
            completed_indices = []
        
        # Formata os choices e values para o CheckboxGroup
        choices = [f"{i+1}. {topic}" for i, topic in enumerate(topics_list)]
        completed_items = [choices[i] for i in completed_indices]
        
        # Atualiza o teaching_plan_progress baseado nos tópicos concluídos
        if len(topics_list) > 0:
            self.state.teaching_plan_progress = len(completed_indices) / len(topics_list)
        else:
            self.state.teaching_plan_progress = 0.0
        
        # Calcula o progresso em porcentagem
        progress_value = self.state.teaching_plan_progress * 100
        
        logger.debug(f"📊 Progresso calculado: {progress_value:.1f}%")
        
        # Atualiza o code_output com o test_code se houver teste ativo
        code_output_value = self.state.test_code if self.state.test_code else ""
        
        # Retorna SEMPRE os 6 valores esperados pelo Gradio
        return (
            "",  # Limpa o input
            history,  # Histórico atualizado
            gr.CheckboxGroup(choices=choices, value=completed_items),  # Checklist atualizado
            progress_value,  # Progresso
            gr.Dataset(samples=self.state.current_examples),  # Exemplos atualizados
            code_output_value  # Código do teste
        )
    
    # Update de exemplos
    def generate_dynamic_examples(self, topic: str = None) -> List[List[str]]:
        """
        Gera exemplos dinâmicos de conversação baseados no contexto atual.
        Se falhar, retorna exemplos padrão.
        """
        try:
            # Instanciação do Agente Generator Examples
            generator_examples_agent = Agent(
                role=self.agent_definitions['generator_examples']['role'],
                goal=self.agent_definitions['generator_examples']['goal'],
                backstory=self.agent_definitions['generator_examples']['backstory'],
                llm=LLM(
                    model="ollama/gpt-oss:120b",
                    base_url=OLLAMA_API_BASE,
                    temperature=0.7
                ),
                verbose=False
            )

            # Instanciação da Tarefa do Generator Examples
            generator_examples_task = Task(
                description=self.task_definitions['generator_examples_task']['description'].format(
                    teaching_plan=self.state.teaching_plan or "Ainda não definido",
                    teaching_plan_progress=self.state.teaching_plan_progress,
                    conversation_history="\n".join(self.state.conversation_history) or "Primeira interação",
                    user_message=self.state.user_message or "Nenhuma mensagem ainda"
                ),
                expected_output=self.task_definitions['generator_examples_task']['expected_output'],
                agent=generator_examples_agent,
                output_pydantic=GeneratorExamplesOutput
            )

            # Execução da Crew
            crew = Crew(
                agents=[generator_examples_agent],
                tasks=[generator_examples_task],
                process=Process.sequential,
                verbose=False
            )
            
            result = crew.kickoff()

            # Parser
            analysis = result.pydantic

            if not analysis or not analysis.examples:
                raise ValueError("O resultado Pydantic do Generator Examples foi None ou vazio.")

            new_examples = analysis.examples
            logger.info(f"✅ Exemplos gerados com sucesso: {len(new_examples)} exemplos")

            return new_examples
            
        except Exception as e:
            logger.warning(f"⚠️ Falha ao gerar exemplos dinamicamente: {e}")
            logger.info("📝 Usando exemplos padrão")
            return self.default_examples
    
    # Atualiza os exemplos dinamicamente
    def update_examples(self):
        """Atualiza os exemplos dinamicamente (pode ser gerado por LLM)."""

        # Gera novos exemplos usando o método da classe
        new_examples = self.generate_dynamic_examples()
        self.state.current_examples = new_examples

        logger.info(f"📝 Exemplos atualizados: {len(new_examples)} novos exemplos")
        return ""
        #return gr.Dataset(samples=new_examples)

    # Callback do exemplo
    def select_example(self, evt: gr.SelectData):
        """Preenche o input quando um exemplo é clicado."""
        # Usa os exemplos padrão se current_examples estiver vazio
        examples = self.state.current_examples if self.state.current_examples else self.default_examples
        
        # Retorna o texto do exemplo selecionado
        if evt.index < len(examples):
            return examples[evt.index][0]
        
        return ""


    ##############################################
    #################### Flow ####################
    ##############################################
    # Fluxo de Execução (Start)
    # Responsável por inicializar o fluxo de execução do DeepMentor.
    # Inicializa a interface Gradio e o estado inicial do fluxo.
    # Envia a instrução inicial para a Interação com o Usuário.
    @start()
    def start_flow(self) -> Any:        
        # Inicializa variáveis se for a primeira vez
        if self.state.turn == 0:
            self.state.user_name = ""
            self.state.agent_response = ""
            self.state.last_agent = ""
            self.state.available_topics = []
            self.state.user_interest_topics = []
            self.state.teaching_plan_progress = 0.0
            self.state.current_examples = self.default_examples
            self.state.next_instruction = "deepmentor presentation"

            logger.info("Estado inicial definido.")
            logger.info("Acionando fluxo de apresentação do DeepMentor.")
            return ""
    
    # Método auxiliar para executar a crew de apresentação
    def _execute_presentation_crew(self) -> bool:
        """
        Executa a crew de apresentação e atualiza o estado.
        Retorna True se todas as informações foram coletadas, False caso contrário.
        """
        logger.info("🎓 Executando crew de apresentação")
        
        # Atualiza o agente atual
        self.state.last_agent = self.agent_definitions['deepmentor_presentation']['role']

        # Instanciação do Agente de Apresentação
        presentation_agent = Agent(
            role=self.agent_definitions['deepmentor_presentation']['role'],
            goal=self.agent_definitions['deepmentor_presentation']['goal'],
            backstory=self.agent_definitions['deepmentor_presentation']['backstory'],
            llm=self.llm,
            #knowledge_sources=[self.d2l_book_knowledge],  # Acesso à base de conhecimento
            verbose=True
        )
        
        # Instanciação da Tarefa de Apresentação
        presentation_task = Task(
            description=self.task_definitions['deepmentor_presentation_task']['description'].format(
                summary_d2l=self.state.summary_d2l,  # Adiciona o summary como parâmetro
                turn=self.state.turn,
                user_name=self.state.user_name or "não informado ainda",
                available_topics=self.state.available_topics if self.state.available_topics else "ainda não consultado",
                user_message=self.state.user_message or "primeira interação",
                conversation_history="\n".join(self.state.conversation_history) if self.state.conversation_history else "primeira interação"
            ),
            expected_output=self.task_definitions['deepmentor_presentation_task']['expected_output'],
            agent=presentation_agent,
            output_pydantic=PresentationOutput
        )
        
        # Log do contexto atual (para debug)
        #logger.info(f"📊 Contexto da conversa:")
        #logger.info(f"   - Turno: {self.state.turn}")
        #logger.info(f"   - Nome: {self.state.user_name or 'não informado'}")
        #logger.info(f"   - Tópicos já listados: {len(self.state.available_topics) if self.state.available_topics else 0}")
        #logger.info(f"   - Última mensagem: {self.state.user_message[:50] if self.state.user_message else 'N/A'}...")
        #logger.info(f"   - Histórico: {len(self.state.conversation_history)} mensagens")
        
        # Execução da Crew
        crew = Crew(
            agents=[presentation_agent],
            tasks=[presentation_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Parser e atualização do estado
        try:
            presentation_output = result.pydantic
            
            if not presentation_output:
                raise ValueError("O resultado Pydantic da Apresentação foi None.")
            
            # Extrai a mensagem de apresentação do objeto pydantic
            self.state.agent_response = presentation_output.presentation_message

            # Atualiza o estado com as informações coletadas
            if presentation_output.user_name:
                self.state.user_name = presentation_output.user_name
                logger.info(f"👤 Nome do usuário identificado: {self.state.user_name}")
            
            if presentation_output.available_topics:
                self.state.available_topics = presentation_output.available_topics
                logger.info(f"📚 Tópicos disponíveis: {len(self.state.available_topics)} tópicos")
            
            if presentation_output.user_interest_topics:
                self.state.user_interest_topics = presentation_output.user_interest_topics
                logger.info(f"💡 Interesses: {', '.join(self.state.user_interest_topics)}")
            
            if presentation_output.user_focus_type:
                self.state.user_focus_type = presentation_output.user_focus_type
                logger.info(f"🎯 Tipo de foco: {self.state.user_focus_type}")
            
            if presentation_output.user_level:
                self.state.user_level = presentation_output.user_level
                logger.info(f"📊 Nível do usuário: {self.state.user_level}")
            
            # Log do status de satisfação
            logger.info(f"📊 Status de coleta:")
            logger.info(f"   - Nome: {presentation_output.user_name_satisfied}")
            logger.info(f"   - Interesses: {presentation_output.user_interest_satisfied}")
            logger.info(f"   - Tema selecionado: {presentation_output.topic_selection_satisfied}")
            logger.info(f"   - Tipo de foco: {presentation_output.user_focus_satisfied}")
            logger.info(f"   - Nível: {presentation_output.user_level_satisfied}")
            logger.info(f"   - Todos requisitos: {'✅' if presentation_output.all_requirements_met else '❌'}")
            
            # Incrementa o turno
            self.state.turn += 1
            
            # Decide próximo passo baseado nos requisitos
            if presentation_output.all_requirements_met:
                logger.info("✅ Todas as informações coletadas → Pronto para próxima fase")
                
                # Salva informações importantes coletadas
                logger.info(f"📝 Informações salvas:")
                logger.info(f"   - Nome: {self.state.user_name}")
                logger.info(f"   - Tópicos de interesse: {self.state.user_interest_topics}")
                logger.info(f"   - Tipo de foco: {self.state.user_focus_type}")
                logger.info(f"   - Nível: {self.state.user_level}")
                logger.info(f"   - Tópicos disponíveis: {len(self.state.available_topics)} tópicos")
                
                # Reinicia o histórico de conversação (como solicitado)
                logger.info("🔄 Reiniciando histórico de conversação para novo ciclo")
                self.state.conversation_history = []
                
                # Informa a última instrução na variável de estado
                self.state.last_instruction = "deepmentor presentation"
                
                # Informa a próxima instrução na variável de estado
                # - encaminha ao orquestrador que a próxima instrução é de "teaching_plan_ordering"
                # - teaching_plan_ordering: responsável por criar o plano de ensino inicial
                self.state.next_instruction = "teaching_plan_ordering"
                self.state.turn = 1  # Reseta o turno para 1 (novo ciclo), ciclo 0 é somente para apresentação

                # Inicia o flow orchestrator
                # - O start_flow não realiza nenhuma ação após o estado inicial e o Crew encaminha o fluxo para o router do orquestrador
                # - Com o valor diferente de "deepmentor presentation", o router do orquestrador encaminha para a rotina de teaching_plan_ordering com o Dean
                self.kickoff()

                return True
            else:
                logger.info("⏸️  Aguardando mais informações do usuário")
                self.state.last_instruction = "deepmentor presentation"
                self.state.next_instruction = "user message"
                return False
                
        except (ValidationError, TypeError, ValueError) as e:
            logger.error(f"❌ Erro ao processar apresentação: {e}")
            logger.error(f"   Resposta bruta: {result.raw}")
            self.state.agent_response = "Desculpe, ocorreu um erro. Pode repetir?"
            return False
    
    # Fluxo de Apresentação (Router):
    @router("deepmentor presentation")
    def deepmentor_presentation(self) -> str:
        """
        Apresentação do DeepMentor e boas-vindas ao usuário.
        - Apresenta o DeepMentor e sua missão
        - Pergunta o nome do usuário
        - Informa os temas disponíveis na base de conhecimento
        - Prepara o terreno para a criação do plano de ensino
        """
        logger.info("🎓 Iniciando apresentação do DeepMentor")
        
        # Executa a crew de apresentação
        all_info_collected = self._execute_presentation_crew()
        
        # Se todas as informações foram coletadas, direciona para o orquestrador
        if all_info_collected:
            logger.info("✅ Redirecionando para orquestrador")
            return "flow orchestrator"
        else:
            # Aguarda próxima mensagem do usuário
            return ""
    
    # Orquestrador de Fluxo:
    @router(
        or_(
            start_flow,
            "flow orchestrator" # condição de retorno do orquestrador de fluxo (id de escuta)
        )
    )
    def flow_orchestrator(self) -> str:
        """
        Orquestrador é responsável por decidir qual o pipeline de execução a seguir.
        """
        # Por padrão: se a instrução for de apresentação, encaminha automaticamente para o fluxo de apresentação
        if self.state.next_instruction == "deepmentor presentation":
            return "deepmentor presentation"
        
        # Se a instrução for qualquer outra, continua o fluxo do orquestrador
        logger.info(f"🎯 Orchestrator - Instrução: {self.state.next_instruction}")

        # Instanciação do Agente Orchestrator
        orchestrator_agent = Agent(
            role=self.agent_definitions['orchestrator']['role'],
            goal=self.agent_definitions['orchestrator']['goal'],
            backstory=self.agent_definitions['orchestrator']['backstory'],
            llm=self.llm
        )

        # Instanciação da Tarefa do Orchestrator
        orchestrator_task = Task(
            description=self.task_definitions['orchestrator_task']['description'].format(
                user_name=self.state.user_name or "não informado",
                user_interest_topics=", ".join(self.state.user_interest_topics) if self.state.user_interest_topics else "não informado",
                user_focus_type=self.state.user_focus_type or "não informado",
                user_level=self.state.user_level or "não informado",
                available_topics=", ".join(self.state.available_topics) if self.state.available_topics else "não consultado",
                turn=self.state.turn,
                last_agent=self.state.last_agent,
                last_instruction=self.state.last_instruction,
                user_message=self.state.user_message or "nenhuma",
                teaching_plan_progress=self.state.teaching_plan_progress,
                teaching_plan=self.state.teaching_plan or "ainda não criado",
                conversation_history="\n".join(self.state.conversation_history) if self.state.conversation_history else "histórico vazio"
            ),
            expected_output=self.task_definitions['orchestrator_task']['expected_output'],
            agent=orchestrator_agent,
            output_pydantic=OrchestratorAnalysis
        )

        # Instanciação da Crew
        crew = Crew(
            agents=[orchestrator_agent],
            tasks=[orchestrator_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()

        # Parser
        try:
            analysis = result.pydantic

            if not analysis:
                raise ValueError("O resultado Pydantic do Orquestrador foi None.")

            # Atualiza o estado com a análise do orquestrador
            self.state.turn += 1
            self.state.next_instruction = analysis.next_instruction
            
            logger.info(f"✅ Orquestrador:")
            logger.info(f"   - Próxima instrução: {self.state.next_instruction}")

            # Roteamento baseado na instrução do orquestrador
            if self.state.next_instruction == "teaching_plan_ordering":
                logger.info("➡️  Direcionando para: Teaching Plan Dean Debate")
                return "teaching plan: dean debate"
            
            # Instruções do Professor (podem vir como "call_professor:*")
            elif self.state.next_instruction.startswith("call_professor:"):
                action = self.state.next_instruction.split(":", 1)[1] if ":" in self.state.next_instruction else ""
                logger.info(f"➡️  Direcionando para Professor: {action}")
                # Atualiza a instrução para o professor processar
                self.state.next_instruction = action
                return "teaching"
            
            # Outras instruções específicas
            elif self.state.next_instruction == "subject_choice":
                logger.info("➡️  Direcionando para: Subject Choice Dean Consult")
                return "subject choice: dean consult"
            
            elif self.state.next_instruction == "start_teaching":
                logger.info("➡️  Direcionando para: Teaching Plan Professor Debate")
                return "teaching plan: professor debate"
            
            elif self.state.next_instruction == "end_session":
                logger.info("✅ Sessão finalizada pelo orquestrador")
                self.state.agent_response = f"Obrigado {self.state.user_name}! Foi um prazer ensinar você. Até a próxima! 👋"
                return ""
            
            else:
                logger.warning(f"⚠️  Instrução não reconhecida: {self.state.next_instruction}")
                logger.info("📝 Finalizando fluxo")
                return ""
            
        except (ValidationError, TypeError, ValueError) as e:
            logger.error(f"❌ O Orquestrador retornou dados inválidos: {e}")
            logger.error(f"   Resposta bruta: {result.raw}")
            return ""
    
        
    # --------------------------------------------------
    # Fluxo 1: Criação e Atualização do Plano de Ensino
    # --------------------------------------------------
    @router("teaching plan: dean debate")
    def teaching_plan_dean_debate(self) -> str:
        # Verifica se é criação ou revisão
        is_revision = self.state.user_feedback != ""
        
        if is_revision:
            logger.info("🔄 Dean: Revisando plano de ensino com feedback do aluno")
            print("dean: teaching plan revision task")
        else:
            logger.info("🎓 Dean: Criando plano de ensino personalizado")
            print("dean: teaching plan task")
        
        try:
            # Extrai o capítulo escolhido pelo usuário
            selected_topic = self.state.user_interest_topics[0] if self.state.user_interest_topics else ""
            
            # Identifica qual capítulo carregar (ex: "Capítulo 3: ..." → "chapter-3")
            chapter_key = None
            for key in self.d2l_data.keys():
                if key != "summary" and selected_topic.lower() in str(self.d2l_data[key]).lower():
                    chapter_key = key
                    break
            
            # Se não encontrou pelo conteúdo, tenta pelo nome
            if not chapter_key:
                import re
                match = re.search(r'capítulo\s+(\d+)|chapter\s+(\d+)', selected_topic.lower())
                if match:
                    chapter_num = match.group(1) or match.group(2)
                    chapter_key = f"chapter-{chapter_num}"
            
            # Carrega o conteúdo do capítulo
            chapter_content = ""
            if chapter_key and chapter_key in self.d2l_data:
                chapter_content = json.dumps(self.d2l_data[chapter_key], ensure_ascii=False, indent=2)
                logger.info(f"📖 Capítulo carregado: {chapter_key}")
                logger.info(f"   Tamanho: {len(chapter_content)} caracteres")
            else:
                logger.warning(f"⚠️ Capítulo não encontrado para: {selected_topic}")
                chapter_content = f"Conteúdo sobre {selected_topic} (capítulo não disponível no momento)"
            
            # Instancia o agente Dean
            dean_agent = Agent(
                role=self.agent_definitions['dean_agent']['role'],
                goal=self.agent_definitions['dean_agent']['goal'],
                backstory=self.agent_definitions['dean_agent']['backstory'],
                llm=self.llm,
                verbose=True
            )
            
            # Instancia a task do Dean
            dean_task = Task(
                description=self.task_definitions['dean_task_teaching_plan_ordering']['description'].format(
                    user_name=self.state.user_name or "Aluno",
                    user_interest_topics=", ".join(self.state.user_interest_topics) if self.state.user_interest_topics else "não especificado",
                    user_focus_type=self.state.user_focus_type or "equilibrado",
                    user_level=self.state.user_level or "iniciante",
                    chapter_content=chapter_content, # Conteúdo completo do capítulo
                    user_feedback=self.state.user_feedback or "Nenhum feedback (primeira vez criando o plano)"
                ),
                expected_output=self.task_definitions['dean_task_teaching_plan_ordering']['expected_output'],
                agent=dean_agent,
                output_pydantic=DeanOutput
            )
            
            # Cria e executa a crew
            crew = Crew(
                agents=[dean_agent],
                tasks=[dean_task],
                process=Process.sequential,
                verbose=True
            )
            
            if is_revision:
                logger.info("🚀 Revisando plano de ensino...")
            else:
                logger.info("🚀 Criando plano de ensino...")
                
            result = crew.kickoff()
            
            # Parser do resultado
            dean_output = result.pydantic
            
            if not dean_output:
                raise ValueError("O resultado Pydantic do Dean foi None.")
            
            # Atualiza o estado com o plano de ensino
            self.state.teaching_plan = dean_output.teaching_plan
            self.state.teaching_plan_progress = dean_output.teaching_plan_progress
            
            # Log do plano criado/revisado
            action = "revisado" if is_revision else "criado"
            logger.info(f"✅ Plano de ensino {action} com {len(self.state.teaching_plan)} tópicos:")
            for i, topic_name in enumerate(self.state.teaching_plan.keys(), 1):
                logger.info(f"   {i}. {topic_name}")
            
            # Cria mensagem para o usuário com formatação correta
            topic_list_md = "\n".join([f"{i}. {topic}" for i, topic in enumerate(self.state.teaching_plan.keys(), 1)])
            
            if is_revision:
                dean_message = dedent(f"""
                    📚 **Plano de Ensino Revisado!**
                    
                    {self.state.user_name}, ajustei o plano conforme seu feedback:
                    
                    **Tópicos do Plano de Ensino:**
                    
                    {topic_list_md}
                    
                    **Configuração:**
                    • Foco: {self.state.user_focus_type}
                    • Nível: {self.state.user_level}
                    
                    **O que você acha agora?** O plano está melhor?
                    
                    Você pode:
                    ✅ Aceitar o plano e começar a aprender
                    🔄 Solicitar mais ajustes (me diga o que gostaria de mudar)
                """).strip()
            else:
                dean_message = dedent(f"""
                    📚 **Plano de Ensino Criado!**
                    
                    Olá {self.state.user_name}! Analisei o {selected_topic} e criei um plano personalizado para você:
                    
                    **Tópicos do Plano de Ensino:**
                    
                    {topic_list_md}
                    
                    **Configuração:**
                    • Foco: {self.state.user_focus_type}
                    • Nível: {self.state.user_level}
                    
                    **O que você acha?** Este plano atende às suas expectativas?
                    
                    Você pode:
                    ✅ Aceitar o plano e começar a aprender
                    🔄 Solicitar ajustes (me diga o que gostaria de mudar)
                """).strip()
            
            self.state.agent_response = dean_message
            self.state.last_agent = "Dean"
            self.state.next_instruction = "teaching_plan_confirmation"
            
            # Limpa o feedback após usar
            if is_revision:
                self.state.user_feedback = ""
            
            logger.info(f"📤 Mensagem do Dean preparada")
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar plano de ensino: {e}")
            import traceback
            traceback.print_exc()
            
            # Mensagem de erro para o usuário
            self.state.agent_response = f"Desculpe, {self.state.user_name}. Ocorreu um erro ao criar o plano de ensino. Por favor, tente novamente."
            self.state.last_agent = "Dean"
        
        return "teaching plan: confirmation"

    # Método auxiliar para executar a crew de confirmação do plano
    def _execute_teaching_plan_confirmation_crew(self) -> bool:
        """
        Executa a crew de confirmação do plano de ensino.
        Retorna True se o plano foi aprovado, False se precisa de revisão.
        """
        logger.info("📋 Executando confirmação do plano de ensino")
        
        # Instanciação do Agente Dean para confirmação
        dean_confirmation_agent = Agent(
            role=self.agent_definitions['dean_agent']['role'],
            goal=self.agent_definitions['dean_agent']['goal'],
            backstory=self.agent_definitions['dean_agent']['backstory'],
            llm=self.llm,
            verbose=True
        )
        
        # Instanciação da Tarefa de Confirmação
        dean_confirmation_task = Task(
            description=self.task_definitions['dean_task_teaching_plan_confirmation']['description'].format(
                user_name=self.state.user_name,
                teaching_plan=json.dumps({k: v for k, v in self.state.teaching_plan.items()}, ensure_ascii=False, indent=2),
                user_message=self.state.user_message,
                conversation_history="\n".join(self.state.conversation_history) if self.state.conversation_history else "primeira confirmação"
            ),
            expected_output=self.task_definitions['dean_task_teaching_plan_confirmation']['expected_output'],
            agent=dean_confirmation_agent,
            output_pydantic=TeachingPlanConfirmationOutput
        )
        
        # Execução da Crew
        crew = Crew(
            agents=[dean_confirmation_agent],
            tasks=[dean_confirmation_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Parser e atualização do estado
        try:
            confirmation_output = result.pydantic
            
            if not confirmation_output:
                raise ValueError("O resultado Pydantic da Confirmação foi None.")
            
            # Atualiza a resposta do agente
            self.state.agent_response = confirmation_output.confirmation_message
            
            # Incrementa o turno
            self.state.turn += 1
            
            if confirmation_output.plan_approved:
                logger.info("✅ Plano de ensino aprovado pelo aluno!")
                self.state.last_instruction = "teaching_plan_confirmation"
                self.state.next_instruction = "start_teaching"
                return True
            else:
                logger.info(f"🔄 Aluno solicitou revisão: {confirmation_output.revision_feedback}")
                self.state.last_instruction = "teaching_plan_confirmation"
                self.state.next_instruction = "teaching_plan_revision"
                # Armazena o feedback para revisão
                self.state.user_feedback = confirmation_output.revision_feedback
                return False
                
        except (ValidationError, TypeError, ValueError) as e:
            logger.error(f"❌ Erro ao processar confirmação: {e}")
            logger.error(f"   Resposta bruta: {result.raw}")
            self.state.agent_response = "Desculpe, ocorreu um erro. Pode repetir se aprova o plano?"
            return False

    @router("teaching plan: confirmation")
    def teaching_plan_confirmation(self) -> str:
        """
        Aguarda a confirmação do usuário sobre o plano de ensino.
        Se aprovado, segue para o ensino. Se não, volta para revisão.
        """
        logger.info("📋 Aguardando confirmação do plano de ensino")
        
        # Executa a crew de confirmação
        plan_approved = self._execute_teaching_plan_confirmation_crew()
        
        if plan_approved:
            logger.info("✅ Redirecionando para início do ensino")
            return "teaching plan: professor debate"
        else:
            # Volta para o Dean revisar o plano
            logger.info("🔄 Redirecionando para revisão do plano")
            return "teaching plan: dean debate"

    
    # ---------------------------------------------
    # 3º Fluxo: Introdução e Exploração de Conceito
    # ---------------------------------------------

    @router(
        or_(
            "teaching plan: professor debate",  # Vem da confirmação do plano
            "teaching"  # Vem do callback quando usuário responde durante ensino
        )
    )
    def teaching_plan_professor_debate(self) -> str:
        logger.info("👨‍🏫 Professor: Iniciando ensino do próximo tópico")
        print("professor: teaching task")
        
        try:
            # Identifica o próximo tópico a ser ensinado (primeiro com valor False)
            current_topic = None
            topic_index = 0
            for i, (topic_name, completed) in enumerate(self.state.teaching_plan.items()):
                if not completed:
                    current_topic = topic_name
                    topic_index = i
                    break
            
            if not current_topic:
                logger.warning("⚠️ Todos os tópicos já foram concluídos!")
                self.state.agent_response = "Parabéns! Você concluiu todos os tópicos do plano de ensino! 🎉"
                self.state.last_agent = "Professor"
                return "teaching plan: gui node"
            
            logger.info(f"📖 Tópico atual: {current_topic}")
            
            # Extrai o capítulo escolhido pelo usuário
            selected_topic = self.state.user_interest_topics[0] if self.state.user_interest_topics else ""
            
            # Identifica qual capítulo carregar
            chapter_key = None
            for key in self.d2l_data.keys():
                if key != "summary" and selected_topic.lower() in str(self.d2l_data[key]).lower():
                    chapter_key = key
                    break
            
            if not chapter_key:
                import re
                match = re.search(r'capítulo\s+(\d+)|chapter\s+(\d+)', selected_topic.lower())
                if match:
                    chapter_num = match.group(1) or match.group(2)
                    chapter_key = f"chapter-{chapter_num}"
            
            # Cria JSONKnowledge do capítulo apontando para o arquivo
            chapter_knowledge = None
            if chapter_key and chapter_key in self.d2l_data:
                # Monta o caminho para o arquivo JSON do capítulo específico
                # chapter_key é algo como "chapter-3", então o arquivo é "chapter-3.json"
                chapter_json_path = f"{chapter_key}.json"
                
                try:
                    chapter_knowledge = JSONKnowledgeSource(
                        file_path=chapter_json_path,
                        metadata={"chapter": chapter_key, "selected_chapter": chapter_key}
                    )
                    logger.info(f"📚 Knowledge carregado do arquivo: {chapter_json_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao carregar knowledge do arquivo: {e}")
                    # Se falhar, continua sem knowledge
                    chapter_knowledge = None
            
            # Instancia o agente Professor com knowledge
            professor_agent = Agent(
                role=self.agent_definitions['professor']['role'],
                goal=self.agent_definitions['professor']['goal'],
                backstory=self.agent_definitions['professor']['backstory'],
                llm=self.llm,
                knowledge_sources=[chapter_knowledge] if chapter_knowledge else [],
                verbose=True
            )
            
            # Calcula o contexto de progresso
            total_topics = len(self.state.teaching_plan)
            completed_topics = sum(1 for v in self.state.teaching_plan.values() if v)
            
            # Instancia a task do Professor
            professor_task = Task(
                description=self.task_definitions['professor_task_teach_topic']['description'].format(
                    user_name=self.state.user_name,
                    current_topic=current_topic,
                    topic_index=topic_index + 1,
                    total_topics=total_topics,
                    user_level=self.state.user_level,
                    user_focus_type=self.state.user_focus_type,
                    teaching_plan=json.dumps({k: v for k, v in self.state.teaching_plan.items()}, ensure_ascii=False, indent=2),
                    conversation_history="\n".join(self.state.conversation_history[-10:]) if self.state.conversation_history else "início do ensino",
                    test_code=self.state.test_code or "Nenhum teste ativo no momento",
                    user_code=self.state.user_code or "Aluno ainda não submeteu código"
                ),
                expected_output=self.task_definitions['professor_task_teach_topic']['expected_output'],
                agent=professor_agent,
                output_pydantic=ProfessorOutput
            )
            
            # Cria e executa a crew
            crew = Crew(
                agents=[professor_agent],
                tasks=[professor_task],
                process=Process.sequential,
                verbose=True
            )
            
            logger.info(f"🚀 Iniciando ensino do tópico: {current_topic}")
            result = crew.kickoff()
            
            # Parser do resultado
            professor_output = result.pydantic
            
            if not professor_output:
                raise ValueError("O resultado Pydantic do Professor foi None.")
            
            # Processa baseado no modo do professor
            mode = professor_output.mode
            logger.info(f"🎯 Modo do Professor: {mode}")
            
            # A mensagem principal sempre vem do message_presentation
            base_message = dedent(f"""
                📖 **Tópico {topic_index + 1}/{total_topics}: {current_topic}**
                
                {professor_output.message_presentation}
            """).strip()
            
            if mode == "teaching":
                # Professor está ensinando
                logger.info(f"📖 Professor está ensinando o tópico")
                
                if professor_output.edu_content:
                    professor_message = dedent(f"""
                        {base_message}
                        
                        ---
                        
                        {professor_output.edu_content}
                        
                        ---
                        
                        **Progresso:** {completed_topics}/{total_topics} tópicos concluídos ({int(self.state.teaching_plan_progress * 100)}%)
                    """).strip()
                else:
                    professor_message = base_message
                
                self.state.edu_content = professor_output.edu_content
                
            elif mode == "testing":
                # Professor está aplicando teste
                logger.info(f"📝 Professor está aplicando teste sobre o tópico")
                
                professor_message = dedent(f"""
                    {base_message}
                    
                    ---
                    
                    **📝 Desafio:**
                    
                    {professor_output.test_content}
                    
                    ---
                    
                    **💻 Use o editor de código ao lado para implementar sua solução.**
                    Quando terminar, clique em "Enviar" para submeter seu código.
                """).strip()
                
                self.state.test_content = professor_output.test_content
                self.state.test_code = professor_output.test_code
                
                # Limpa o código do usuário (novo teste)
                self.state.user_code = ""
                
            elif mode == "evaluating":
                # Professor está avaliando
                logger.info(f"✅ Professor está avaliando a resposta do aluno")
                
                professor_message = dedent(f"""
                    {base_message}
                    
                    ---
                    
                    {professor_output.result}
                    
                    ---
                    
                    **Progresso:** {completed_topics}/{total_topics} tópicos concluídos ({int(self.state.teaching_plan_progress * 100)}%)
                """).strip()
                
                self.state.result = professor_output.result
                
                # Se a avaliação foi positiva, marca o tópico como concluído
                if "aprovado" in professor_output.result.lower() or "correto" in professor_output.result.lower():
                    self.state.teaching_plan[current_topic] = True
                    completed_topics += 1
                    self.state.teaching_plan_progress = completed_topics / total_topics
                    logger.info(f"✅ Tópico '{current_topic}' concluído!")
                    logger.info(f"📊 Progresso: {int(self.state.teaching_plan_progress * 100)}%")
                    
                    # Limpa os códigos após aprovação
                    self.state.test_code = ""
                    self.state.user_code = ""
            else:
                logger.warning(f"⚠️ Modo desconhecido: {mode}")
                professor_message = base_message
            
            self.state.agent_response = professor_message
            self.state.last_agent = "Professor"
            self.state.last_instruction = "continue_teaching"
            self.state.next_instruction = "user_message"  # ✅ Aguarda mensagem do usuário
            
            logger.info(f"📤 Conteúdo do professor preparado (modo: {mode})")
            logger.info(f"⏸️  Aguardando próxima mensagem do usuário...")
            
        except Exception as e:
            logger.error(f"❌ Erro ao ensinar tópico: {e}")
            import traceback
            traceback.print_exc()
            
            # Mensagem de erro para o usuário
            self.state.agent_response = f"Desculpe, {self.state.user_name}. Ocorreu um erro ao ensinar o tópico. Por favor, tente novamente."
            self.state.last_agent = "Professor"
            self.state.next_instruction = "user_message"
        
        # ✅ Retorna vazio para sair do flow e voltar ao chat
        return ""

    # -----------------------------------------------------
    # TODO: 2º Fluxo: Verificação do cumprimento do Teaching Plan
    # -----------------------------------------------------