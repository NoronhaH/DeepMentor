import os
import subprocess
import sys
import platform
from pathlib import Path

def create_project_structure(
    flow_name="deepmentor_flow",
    crew_name="teaching_crew"
):
    """
    Creates the combined project structure (CCDS and CrewAI)
    and generates the configuration files.
    """
    print(f"Inicializando a estrutura do projeto para: {flow_name}")

    # Define the root directory as the current location
    proj_root = Path.cwd()
    print(f"Raiz do projeto: {proj_root}")

    # CrewAI structure
    flow_root  = proj_root / "flows" / flow_name 
    crew_dir   = flow_root / "crews" / crew_name 
    tools_dir  = flow_root / "tools"             # flow-specific tools
    config_dir = crew_dir / "config"

    # CCDS (Cookiecutter Data Science structure)
    print("\nVerificando a estrutura do repositório Cookiecutter Data Science...")

    # CCDS structure
    repo_structure = [     
        "data/raw",
        "data/interim",
        "data/processed",
        "data/external",
        "deepmentor",
        "models",
        "notebooks",
        f"{crew_dir}/knowledge",
        f"{crew_dir}/short_term_memory",
        f"{crew_dir}/long_term_memory",
        f"{crew_dir}/entities"
    ]

    for d in repo_structure:
        dir_path = proj_root / d 
        
        if not dir_path.is_dir():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✔️  Diretório {d} criado.")
        else:
            print(f"✔️  Diretório {d} verificado.")
    
    # Verify CrewAI repo structure
    print("\nVerificando a estrutura do repositório CrewAI...")

    dirs_crewai = [flow_root, crew_dir, tools_dir, config_dir]

    for d in dirs_crewai:
        if not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            print(f"✔️  Diretório {d.relative_to(proj_root)} criado.")
        else:
            print(f"✔️  Diretório {d.relative_to(proj_root)} verificado.")

    # Placeholder files
    print("\nVerificando/Criando arquivos placeholder...")

    crewai_files = {
        config_dir / "agents.yaml": "# Definições dos agentes para o teaching crew",
        config_dir / "tasks.yaml": "# Definições das tarefas para o teaching crew",
        crew_dir / f"{crew_name}.py": f"# Script principal para o '{crew_name}'\n\n",
        tools_dir / "custom_tool.py": "# Implementação das ferramentas customizadas (ex: OCR, etc.)\n\n",
        flow_root / "README.md": f"# {flow_name.replace('_', ' ').title()}\n\nFluxo do Tutor Adaptativo de Deep Learning.",
        flow_root / "pyproject.toml": "# Configuração do projeto (alternativa ao requirements.txt)\n\n"
    }

    for f_path, f_content in crewai_files.items():
        if not f_path.exists():
            f_path.write_text(f_content)
            print(f"✔️  Arquivo {f_path.relative_to(proj_root)} criado.")
        else:
            print(f"✔️  Arquivo {f_path.relative_to(proj_root)} verificado.")

    print("\n✅ Estrutura do projeto verificada/criada com sucesso!")

def run_command(command, shell=False):
    """
    Runs a shell command and stops the script on error.
    """
    try:
        if shell:
            print(f"Executando (shell): {command}")
            subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
        else:
            print(f"Executando: {' '.join(command)}")
            subprocess.run(
                command,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar comando: {e.cmd}")
        print(f"Erro: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Erro: Comando '{command[0]}' não encontrado. Está instalado e no PATH?")
        sys.exit(1)

def install_dependencies(venv_path):
    """
    Installs system (apt) and Python (pip) dependencies.
    """
    print("\nIniciando Instalação das Dependências")

    # System Dependencies (apt)
    if platform.system() == "Linux":
        print("\nLinux detectado. Tentando instalar 'poppler-utils' via apt...")
        try:
            run_command(['sudo', 'apt', 'update', '-y'])
            run_command(['sudo', 'apt', 'install', 'poppler-utils', '-y'])
            print("✔️  'poppler-utils' instalado com sucesso.")
        except Exception as e:
            print(f"\nAVISO: Falha ao instalar 'poppler-utils' via apt. {e}")
            print("Pode ser necessário instalar manually. (ex: 'sudo apt install poppler-utils')")
    else:
        print(f"\nAVISO: Sistema operacional '{platform.system()}' detectado.")
        print("Pulando instalação do 'poppler-utils' (apt). Instale manualmente se necessário.")

    # Python Dependencies (pip)
    print("\nInstalando pacotes Python do 'requirements.txt'...")

    # Specifies the path to the Python executable inside the venv
    if platform.system() == "Windows":
        python_executable = venv_path / "Scripts" / "python.exe"
    else:
        python_executable = venv_path / "bin" / "python"

    # Verifies if the requirements.txt exists
    req_file = Path.cwd() / "requirements.txt"
    if not req_file.exists():
        print(f"Erro: 'requirements.txt' não encontrado em {Path.cwd()}")
        print("Por favor, crie o 'requirements.txt' antes de executar este setup.")
        sys.exit(1)

    pip_command = [
        str(python_executable),
        "-m", "pip",
        "install",
        "-U",
        "-r", "requirements.txt"
    ]
    
    run_command(pip_command)

    print("\n✅ Instalação das dependências concluída!")

if __name__ == "__main__":
    # Setup checker
    flow_name = "deepmentor_flow"
    crew_name = "teaching_crew"

    create_project_structure(
        flow_name=flow_name,
        crew_name=crew_name
    )

    print("\nConfigurando Ambiente Virtual")
    venv_path = Path.cwd() / ".venv"
    if not venv_path.exists():
        print("Criando ambiente virtual '.venv'...")
        run_command([
            sys.executable, '-m', 'venv', str(venv_path)
        ])
    else:
        print("Ambiente virtual '.venv' já existe.")

    install_dependencies(venv_path=venv_path)

    print("\n🎉 Setup do projeto DeepMentor concluído!")
    print("\nPróximos Passos")
    print("1. Ative o ambiente virtual:")
    print("   No Linux/macOS: source .venv/bin/activate")
    print("   No Windows:      .venv\\Scripts\\activate")
    print("2. Comece a utilizar o Deep Mentor!")