# Guia para Agentes (Codex)

Este arquivo fornece instruções ao agente Codex sobre como executar, testar e interagir com o projeto **DevAI-R1**.

## Comandos de Configuração e Execução

- **Instalar dependências**: `pip install -r requirements.txt`
  (Instala todas as bibliotecas Python necessárias listadas em requirements.txt)
- **Instalar ferramentas de desenvolvimento**: `pip install -r requirements-dev.txt`
  (Instala flake8, pylint, mypy, bandit, pre-commit e demais utilitários.)

- **Executar todos os testes**: `pytest`
  (Roda a suíte completa de testes automatizados. Todos os testes devem passar antes de considerar uma tarefa concluída.)

- **Analisar estilo de código**: `flake8 devai`  
  (Verifica o estilo do código conforme as convenções do projeto.)

- **Análise estática completa**: `pylint devai && mypy devai`  
  (Executa o Pylint para detectar possíveis problemas e o MyPy para checar consistência de tipos estáticos.)

- **Análise de segurança**: `bandit -r devai`
  (Realiza varredura de segurança no código em busca de vulnerabilidades conhecidas.)

### Atalhos úteis

- **Checar tudo de uma vez**: `./scripts/dev_checks.sh`
  (Roda flake8, pylint, mypy, bandit e pytest em sequência.)

## Convenções do Projeto

- Siga as recomendações do PEP8 para formatação de código Python (já verificado pelo flake8/pylint).
- Garantir que todas as funções e métodos importantes possuam docstrings explicativas.
- O projeto utiliza contexto de memória via FAISS; evite modificar arquivos de índice vetorial sem necessidade.
- Tokens de API e credenciais **não** devem ser expostos no código ou logs. Use variáveis de ambiente (como `OPENROUTER_API_KEY`) para acessar serviços externos.
- Após implementar uma mudança, sempre rode os testes (`pytest`) para validar que nada foi quebrado. O agente deve priorizar solucionar quaisquer falhas nos testes antes de concluir a tarefa.
- Dê preferência a `pre-commit run --files <arquivos>` antes de enviar commits para garantir que o estilo e as verificações básicas estão corretos.

## Localizando e Corrigindo Falhas

Quando a suíte completa é grande, use estes comandos para focar nos erros:

- `pytest -k <expressao>` executa apenas testes que contenham a expressão no nome.
- `pytest tests/caminho/test_mod.py::TestClasse::test_alvo -vv` roda um teste específico com saída detalhada.
- `pytest --maxfail=1 -vv` interrompe na primeira falha, útil para ciclos rápidos.
- `pytest --lf -vv` executa somente os testes que falharam na execução anterior.

Use `grep -Rn "pattern"` para localizar trechos problemáticos no código ou `logs/` para consultar registros de execução. Após corrigir, execute novamente `pytest` para validar.

## Estrutura do Projeto (resumo rápido)

- `devai/` – Pacote principal com os módulos do assistente (core, memória, análise, APIs, etc).
- `tests/` – Testes automatizados cobrindo funcionalidades chave.
- `DevAI_R1.py` – Script/entrada principal do assistente (possível interface CLI/TUI).
- ... *(adicione breves descrições se necessário dos outros diretórios ou arquivos importantes).* ...

Para detalhes extras consulte também `README.md`, `PROJECT_GUIDE.md` e os documentos em `docs/`.

## Resolução e Registro de Erros

Sempre que manipular o projeto, siga o fluxo abaixo:

1. Rode os testes (`pytest`) e as verificações de estilo (`./scripts/dev_checks.sh` ou comandos individuais).
2. Caso alguma falha seja encontrada, tente corrigi-la imediatamente. Priorize deixar a suíte de testes passando.
3. Se algum erro persistir e não for possível resolvê-lo na mesma tarefa, registre as informações no arquivo `unresolved_errors.jsonl` no formato JSONL, incluindo `timestamp`, `tipo`, `mensagem` e `funcao`/`arquivo` relacionado.
4. Mantenha esse log versionado para que outros desenvolvedores saibam o que ainda precisa ser tratado.
