# Guia do Projeto DevAI

Este arquivo permite descrever comandos e convenções específicas do projeto. O agente utiliza estas informações para compilar e testar corretamente.

## Comandos principais

- **Instalar dependências:**
  `pip install -r requirements-core.txt && pip install -r requirements-ml.txt && pip install -r requirements-ui.txt`
- **Executar testes:** `pytest`
- **Verificar estilo:** `flake8 devai`
- **Análise estática:** `pylint devai && mypy devai`
- **Análise de segurança:** `bandit -r devai`

Todos esses passos podem ser executados em sequência utilizando o script `./scripts/dev_checks.sh`.

Antes de enviar contribuições, recomendamos rodar `pre-commit run --files <arquivos>` localmente para garantir que formatação, lint, tipos e testes estejam corretos.

## Passos adicionais

- Para a interface web opcional, instale as dependências JavaScript em `static/`:
  `npm install --prefix static`
- Caso deseje executar os testes em isolamento, certifique-se de ter o Docker
  instalado e habilite `TESTS_USE_ISOLATION: true` no `config.yaml`. O DevAI
  irá rodar `pytest` dentro de um container conforme os limites configurados.

## Plugins e tarefas

O gerenciador de plugins permite adicionar novas tarefas ao sistema. Para
ativar um plugin ou executar tarefas manualmente, inicie a CLI:

```bash
python -m devai --cli
/plugins               # lista o status dos plugins
/plugin todo_counter on
/tarefa run_tests
/tarefa static_analysis
```

Mantenha estas instruções em sincronia com o `README.md` para evitar divergências.
