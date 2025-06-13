# Guia para Agentes (Codex)

Este arquivo fornece instruções ao agente Codex sobre como executar, testar e interagir com o projeto **DevAI-R1**.

## Comandos de Configuração e Execução

- **Instalar dependências**: `pip install -r requirements.txt`  
  (Instala todas as bibliotecas Python necessárias listadas em requirements.txt)

- **Executar todos os testes**: `pytest`  
  (Roda a suíte completa de testes automatizados. Todos os testes devem passar antes de considerar uma tarefa concluída.)

- **Analisar estilo de código**: `flake8 devai`  
  (Verifica o estilo do código conforme as convenções do projeto.)

- **Análise estática completa**: `pylint devai && mypy devai`  
  (Executa o Pylint para detectar possíveis problemas e o MyPy para checar consistência de tipos estáticos.)

- **Análise de segurança**: `bandit -r devai`  
  (Realiza varredura de segurança no código em busca de vulnerabilidades conhecidas.)

## Convenções do Projeto

- Siga as recomendações do PEP8 para formatação de código Python (já verificado pelo flake8/pylint).
- Garantir que todas as funções e métodos importantes possuam docstrings explicativas.
- O projeto utiliza contexto de memória via FAISS; evite modificar arquivos de índice vetorial sem necessidade.
- Tokens de API e credenciais **não** devem ser expostos no código ou logs. Use variáveis de ambiente (como `OPENROUTER_API_KEY`) para acessar serviços externos.
- Após implementar uma mudança, sempre rode os testes (`pytest`) para validar que nada foi quebrado. O agente deve priorizar solucionar quaisquer falhas nos testes antes de concluir a tarefa.

## Estrutura do Projeto (resumo rápido)

- `devai/` – Pacote principal com os módulos do assistente (core, memória, análise, APIs, etc).
- `tests/` – Testes automatizados cobrindo funcionalidades chave.
- `DevAI_R1.py` – Script/entrada principal do assistente (possível interface CLI/TUI).
- ... *(adicione breves descrições se necessário dos outros diretórios ou arquivos importantes).* ...
