# DeepMentor 🎓

> **Disciplina:** IA368HH - FEEC-UNICAMP  
> **Responsável pelo projeto:** Hiuri Noronha
> **Supervisão:** Prof. Roberto Lotufo

## Visão Geral

O **DeepMentor** é um sistema inteligente de tutoria adaptativa que utiliza uma arquitetura multi-agente baseada em **CrewAI Flows** para promover aprendizado personalizado em Deep Learning. Diferentemente de chatbots convencionais baseados apenas em Retrieval-Augmented Generation (RAG), o DeepMentor é capaz de:

- 🎯 **Planejar trajetórias de ensino personalizadas** baseadas no perfil do aluno
- 📊 **Avaliar dinamicamente** o nível de compreensão através de testes práticos
- 🔄 **Ajustar o conteúdo em tempo real** conforme o progresso do aprendiz
- 💡 **Ensinar de forma socrática** com ciclos de ensino-teste-avaliação

### Base de Conhecimento

O sistema utiliza como fonte o livro **"Dive into Deep Learning"** (Zhang et al., 2023), processado via OCR e estruturado em formato JSON para acesso eficiente pelos agentes.

## Arquitetura Multi-Agente

O DeepMentor implementa uma arquitetura de fluxo orquestrado com 5 agentes especializados:

### 1. **DeepMentor Presentation Agent** 👋
- **Função:** Primeiro contato com o aluno, coleta de perfil
- **Responsabilidades:**
  - Apresentar o sistema e a base de conhecimento
  - Coletar nome, interesses, nível (iniciante/intermediário/avançado)
  - Identificar tipo de foco (teórico/prático/equilibrado)
  - Listar tópicos disponíveis e capturar seleção do aluno

### 2. **Orchestrator Agent** 🎯
- **Função:** Coordenador central de fluxos
- **Responsabilidades:**
  - Analisar o estado atual da interação (turno, agente anterior, instrução)
  - Decidir qual pipeline executar a seguir
  - Rotear entre os agentes Dean e Professor conforme o contexto
  - Gerenciar transições entre as fases do fluxo

### 3. **Dean Agent** 📚
- **Função:** Diretor Acadêmico e Arquiteto de Planos de Ensino
- **Responsabilidades:**
  - Criar plano de ensino personalizado (4-8 tópicos progressivos)
  - Analisar o conteúdo do capítulo selecionado
  - Adaptar profundidade conforme o nível do aluno
  - Gerenciar aprovação/revisão do plano pelo aluno
  - Atualizar progresso conforme tópicos são concluídos

### 4. **Professor Agent** 👨‍🏫
- **Função:** Tutor Especialista em Deep Learning
- **Responsabilidades:**
  - Ensinar tópicos de forma didática e adaptativa
  - Criar testes práticos de código (Python)
  - Avaliar respostas e código submetido pelo aluno
  - Fornecer feedback construtivo e específico
  - Seguir ciclo pedagógico: **ENSINAR → TESTAR → AVALIAR**

### 5. **Generator Examples Agent** 💡
- **Função:** Gerador de Sugestões Contextuais
- **Responsabilidades:**
  - Gerar exemplos dinâmicos de mensagens baseados no contexto
  - Atualizar sugestões conforme o progresso do aluno
  - Facilitar a interação do usuário

## Fluxos Implementados

### **Fase 1: Apresentação e Coleta de Perfil**
1. Sistema apresenta o DeepMentor e base de conhecimento D2L
2. Coleta informações do aluno:
   - Nome
   - Tópicos de interesse
   - Tema/capítulo escolhido
   - Tipo de foco (teórico/prático/equilibrado)
   - Nível (iniciante/intermediário/avançado)
3. Valida se todas as informações foram coletadas

### **Fase 2: Criação e Confirmação do Plano de Ensino**
1. Dean analisa o capítulo selecionado
2. Cria plano de ensino personalizado (4-8 tópicos)
3. Apresenta o plano ao aluno
4. Aguarda confirmação ou solicitação de ajustes
5. Revisa o plano conforme feedback (se necessário)

### **Fase 3: Ensino Iterativo**
Para cada tópico do plano:
1. **Ensino:** Professor apresenta o conteúdo de forma didática
2. **Esclarecimento:** Responde dúvidas do aluno
3. **Teste:** Cria desafio prático de código com template
4. **Avaliação:** Analisa código submetido e fornece feedback
5. **Progressão:** Marca tópico como concluído se aprovado
6. **Repetição:** Avança para próximo tópico

```
deepmentor/
├── deepmentor/                      # Módulo principal do projeto
│   ├── __init__.py
│   ├── config.py                    # Configurações centralizadas (paths, APIs, modelos)
│   ├── main.py                      # Ponto de entrada principal
│   ├── data_loader.py               # Ferramentas de processamento da base de conhecimento
│   └── tester.py                    # Scripts de teste de APIs
│
├── flows/                           # Fluxos CrewAI
│   └── deepmentor_flow/
│       ├── crews/
│       │   └── teaching_crew/
│       │       ├── config/
│       │       │   ├── agents.yaml  # Definição dos agentes
│       │       │   └── tasks.yaml   # Definição das tarefas
│       │       ├── knowledge/       # Base de conhecimento estática
│       │       │   ├── d2l-ocr.json
│       │       │   └── chapter-3.json
│       │       └── teaching_crew.py # Implementação do fluxo multi-agente
│       └── tools/                   # Ferramentas customizadas (futuro)
│
├── knowledge/                       # Base de conhecimento dinâmica (runtime)
│   └── d2l-ocr.json                # Copiado em runtime do knowledge estático
│
├── data/                            # Dados do projeto
│   ├── raw/
│   │   └── d2l_book/
│   │       ├── d2l-en.pdf          # Livro original
│   │       └── pages/              # Páginas convertidas em PNG
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── docs/                            # Documentação gerada
│   ├── DeepMentorFlow.html         # Visualização do fluxo CrewAI
│   ├── crewai_flow_script.js
│   └── crewai_flow_style.css
│
├── models/                          # Modelos treinados (futuro)
├── notebooks/                       # Jupyter notebooks para análises
├── reports/                         # Relatórios e figuras
│
├── requirements.txt                 # Dependências Python
├── .env                            # Variáveis de ambiente (não versionado)
└── README.md                       # Este arquivo
```

## Instalação e Configuração

### 1. **Pré-requisitos**
- Python 3.10+
- Ollama (opcional, para modelos locais)
- Acesso à API OpenAI (recomendado)

### 2. **Clone o Repositório**
```bash
git clone <repository-url>
cd deepmentor
```

### 3. **Crie e Ative o Ambiente Virtual**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

### 4. **Instale as Dependências**
```bash
pip install -U -r requirements.txt
```

### 5. **Configure as Variáveis de Ambiente**
Crie um arquivo `.env` na raiz do projeto:

```env
# APIs de LLM
OPENAI_API_KEY=sk-your-openai-key-here
OLLAMA_API_BASE=http://localhost:11434

# API de OCR (NeuralMind) - Opcional
NEURALMIND_API_KEY=your-neuralmind-key
NEURALMIND_OCR_URL=https://api.neuralmind.ai/ocr
```

### 6. **Configure o Modelo no `config.py`**
Edite `deepmentor/config.py`:

```python
# Escolha do modelo
LLM_MODEL = 0  # 0 = OpenAI, 1 = Ollama

# Modelos específicos
GPT_MODEL = "gpt-4o-mini"          # Para OpenAI
OLLAMA_MODEL = "ollama/gpt-oss:120b"  # Para Ollama
```

## Preparação da Base de Conhecimento

### Opção 1: Usar Base Pré-processada (Recomendado)
Se você já possui os arquivos `d2l-ocr.json` e `chapter-3.json` na pasta `flows/deepmentor_flow/crews/teaching_crew/knowledge/`, pule para a seção "Execução".

### Opção 2: Processar do Zero
Execute os seguintes comandos para processar o livro D2L:

```bash
# 1. Baixar o livro "Dive into Deep Learning"
python3 -m deepmentor.data_loader download-book

# 2. Converter páginas do PDF para PNG
python3 -m deepmentor.data_loader prepare-images

# 3. Executar OCR nas páginas (requer API NeuralMind)
python3 -m deepmentor.data_loader run-ocr

# 4. Calcular tokens do conteúdo processado
python3 -m deepmentor.data_loader calculate-tokens
```

## Execução

### Iniciar o DeepMentor
```bash
python3 -m deepmentor.main
```

Isso irá:
1. Carregar a base de conhecimento
2. Inicializar os agentes
3. Gerar visualização do fluxo em `docs/DeepMentorFlow.html`
4. Iniciar a interface Gradio

### Acessar a Interface
Após a execução, acesse:
- **Local:** `http://localhost:7860`
- **Público (se habilitado):** Link exibido no terminal

### Testar APIs (Opcional)
```bash
# Testar conexão com Ollama
python3 -m deepmentor.tester --test_self_hosted

# Testar conexão com OpenAI
python3 -m deepmentor.tester --test_openai
```

## Exemplo de Uso

1. **Início da Conversa:**
   - Sistema: "Olá! Sou o DeepMentor 🎓. Qual é seu nome?"
   - Aluno: "Meu nome é João"

2. **Coleta de Perfil:**
   - Sistema: Lista capítulos disponíveis do D2L
   - Aluno: "Quero aprender Capítulo 3: Regressão Linear"
   - Sistema: "Qual seu nível e tipo de foco?"
   - Aluno: "Iniciante e foco equilibrado"

3. **Plano de Ensino:**
   - Dean apresenta plano com 5 tópicos progressivos
   - Aluno aprova ou solicita ajustes

4. **Ensino:**
   - Professor ensina primeiro tópico
   - Esclarece dúvidas
   - Aplica teste prático de código
   - Avalia resposta
   - Avança para próximo tópico

## Tecnologias Utilizadas

- **CrewAI:** Framework multi-agente com suporte a Flows
- **Gradio:** Interface web interativa
- **Pydantic:** Validação de dados estruturados
- **LangChain/OpenAI:** Integração com LLMs
- **Ollama:** Suporte a modelos locais
- **tiktoken:** Contagem de tokens
- **PyPDF2/pdf2image:** Processamento de PDFs
- **Loguru:** Logging avançado

## Funcionalidades Implementadas

- ✅ Coleta de perfil do aluno (nome, nível, foco, interesses)
- ✅ Criação automática de plano de ensino personalizado
- ✅ Confirmação e revisão do plano pelo aluno
- ✅ Ensino adaptativo por tópicos
- ✅ Testes práticos de código Python
- ✅ Avaliação automática com feedback construtivo (incompleto)
- ✅ Acompanhamento de progresso (checklist + barra de progresso) (incompleto)
- ✅ Interface Gradio com chat + editor de código
- ✅ Geração de exemplos contextuais dinâmicos
- ✅ Visualização do fluxo de agentes
- ✅ Base de conhecimento RAG com JSONKnowledgeSource

## Funcionalidades Futuras

- 🔲 Histórico de sessões persistente
- 🔲 Múltiplos capítulos simultâneos
- 🔲 Execução e validação automática de código
- 🔲 Métricas de aprendizado (tempo, tentativas, scores)
- 🔲 Exportação de progresso em PDF
- 🔲 Suporte a múltiplos idiomas
- 🔲 Integração com Jupyter Notebooks
- 🔲 Sistema de badges e gamificação

## Fases de Desenvolvimento

### ✅ Fase 1: Infraestrutura e Base de Conhecimento (Concluída)
- Estrutura de diretórios baseada em Cookiecutter Data Science
- Sistema de configuração centralizado (`config.py`)
- Pipeline de processamento do livro D2L:
  - Download automático
  - Conversão PDF → PNG
  - OCR via API NeuralMind
  - Cálculo de tokens com tiktoken
- Base de conhecimento estruturada em JSON

### ✅ Fase 2: Arquitetura Multi-Agente (Concluída)
- Implementação do framework CrewAI com Flows
- Definição de 5 agentes especializados:
  - DeepMentor Presentation
  - Orchestrator
  - Dean
  - Professor
  - Generator Examples
- Sistema de roteamento entre agentes
- Validação de saídas com Pydantic

### ✅ Fase 3: Fluxos de Interação (Concluída)
- Fluxo de apresentação e coleta de perfil
- Fluxo de criação e confirmação de plano de ensino
- Fluxo de ensino iterativo (ensinar → testar → avaliar)
- Sistema de gerenciamento de estado (InteractionState)

### ✅ Fase 4: Interface de Usuário (Concluída)
- Interface Gradio com chat interativo
- Editor de código para testes práticos
- Checklist de progresso do plano de ensino
- Barra de progresso visual
- Sistema de sugestões contextuais dinâmicas
- Visualização do fluxo de agentes (HTML)

### 🔄 Fase 5: Refinamento e Otimização (TODO)
- Ajuste de prompts para melhor performance
- Otimização de tokens e custos de API
- Melhoria na avaliação de código
- Testes de usabilidade

### 📋 Fase 6: Expansão (TODO)
- Suporte a múltiplos capítulos
- Sistema de persistência de sessões
- Métricas e analytics de aprendizado
- Execução automática de código Python
- Gamificação e badges

## Configurações Avançadas

### Ajuste de Modelos
Edite `deepmentor/config.py`:

```python
# Para usar GPT-4
GPT_MODEL = "gpt-4"
LLM_MODEL = 0

# Para usar Ollama local
OLLAMA_MODEL = "ollama/llama3:70b"
LLM_MODEL = 1

# Para interface pública Gradio
GRADIO_PUBLIC_SHARE = True
```

### Customizar Capítulos para OCR
Edite a lista `OCR_PAGES` em `config.py`:

```python
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
    },
    {
        "section": "chapter-4",  # Novo capítulo
        "start_page": 165,
        "end": 210
    }
]
```

### Logs e Debug
O sistema usa **Loguru** para logging. Para ajustar verbosidade:

```python
# Em config.py
FLOW_VERBOSITY = True  # False para logs mínimos
```

## Solução de Problemas

### Erro: "OPENAI_API_KEY não encontrada"
- Verifique se o arquivo `.env` existe na raiz do projeto
- Confirme que a variável `OPENAI_API_KEY` está definida no `.env`

### Erro: "Não foi possível importar os módulos"
```bash
# Reinstale as dependências
pip install -U -r requirements.txt
```

### Erro: "Knowledge not found"
```bash
# Verifique se os arquivos JSON existem
ls flows/deepmentor_flow/crews/teaching_crew/knowledge/
# Deve conter: d2l-ocr.json e chapter-3.json
```

### Interface Gradio não abre
- Verifique se a porta 7860 está disponível
- Tente acessar manualmente: `http://127.0.0.1:7860`
- Habilite compartilhamento público: `GRADIO_PUBLIC_SHARE = True` no `config.py`

## Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## Licença

Este projeto é desenvolvido como parte da disciplina IA368HH da FEEC-UNICAMP.

---

## Estrutura da Base de Conhecimento

A base de conhecimento é estruturada em JSON com a seguinte hierarquia:

```json
{
  "summary": {
    "page_number": {
      "text": "Conteúdo OCR da página"
    }
  },
  "chapter-X": {
    "page_number": {
      "text": "Conteúdo OCR da página do capítulo"
    }
  }
}
```

### Capítulos Processados (Atual)
- **Summary:** Páginas 3-24 (índice, prefácio, visão geral)
- **Chapter 3:** Páginas 122-164 (Regressão Linear)

### Como os Agentes Usam o Conhecimento

1. **DeepMentor Presentation:** Lê o `summary` para listar capítulos disponíveis
2. **Dean:** Analisa o conteúdo completo do capítulo escolhido para criar o plano
3. **Professor:** Usa `JSONKnowledgeSource` para buscar informações específicas durante o ensino
4. **Todos os agentes:** Acessam via `self.state.summary_d2l` e `self.state.chapter_3_d2l`

## Performance e Custos

### Contagem de Tokens (Base de Conhecimento)
- **Summary:** ~15.000-20.000 tokens
- **Chapter 3:** ~40.000-50.000 tokens
- **Total:** ~60.000-70.000 tokens

### Custo Estimado por Sessão (GPT-4o-mini)
- Coleta de perfil: ~2.000 tokens ($0.001)
- Criação do plano: ~50.000 tokens ($0.025)
- Ensino por tópico: ~10.000 tokens ($0.005)
- **Sessão completa (5 tópicos):** ~100.000 tokens (~$0.05)

*Valores aproximados, variam conforme o modelo e verbosidade*

## Arquitetura Técnica

### Padrões de Design Utilizados

- **State Machine:** Gerenciamento de estado com `InteractionState`
- **Router Pattern:** Fluxo condicional entre agentes
- **Strategy Pattern:** Agentes especializados com responsabilidades únicas
- **Observer Pattern:** Atualização reativa da interface Gradio
- **Factory Pattern:** Criação de agentes e tarefas via YAML

### Fluxo de Dados

```
Usuario → Gradio → chat_callback() → State Update → Flow Router
                                                        ↓
                        ┌──────────────────────────────────────┐
                        │         Orchestrator                 │
                        └──────────────────────────────────────┘
                                ↓                ↓              
                        ┌──────────┐    ┌─────────────┐        
                        │   Dean   │    │  Professor  │        
                        └──────────┘    └─────────────┘        
                                ↓                ↓              
                        Knowledge Base      JSONKnowledge      
                                ↓                ↓              
                        State Update ← Response Format         
                                ↓                               
                        Gradio Interface Update                
```

## Troubleshooting Avançado

### Agentes não respondem adequadamente
- Verifique os prompts em `config/tasks.yaml`
- Ajuste a temperatura do modelo no `config.py`
- Ative verbose nos agentes: `verbose: true` no `agents.yaml`

### Erro de validação Pydantic
- Modelos LLM podem retornar JSON mal formatado
- Ajuste o prompt para reforçar a estrutura JSON
- Use modelos mais avançados (GPT-4 em vez de GPT-3.5)

### Base de conhecimento muito grande
- Reduza o número de páginas em `OCR_PAGES`
- Use embedding + vector search (implementação futura)
- Fragmente o conhecimento por seção

## Referências

### Base de Conhecimento

**Dive into Deep Learning** - Zhang et al. (2023)

```bibtex
@book{zhang2023dive,
    title={Dive into Deep Learning},
    author={Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J.},
    publisher={Cambridge University Press},
    note={\url{https://D2L.ai}},
    year={2023}
}
```

### Frameworks e Tecnologias

- **CrewAI:** [https://www.crewai.com/](https://www.crewai.com/)
- **Gradio:** [https://gradio.app/](https://gradio.app/)
- **Pydantic:** [https://docs.pydantic.dev/](https://docs.pydantic.dev/)

## Contato

Para dúvidas, sugestões ou contribuições:

- **Responsável:** Hiuri Noronha
- **Disciplina:** IA368HH - FEEC-UNICAMP
- **Ano:** 2025

---

**DeepMentor** - Tutoria Adaptativa de Deep Learning com Arquitetura Multi-Agente 🎓🤖