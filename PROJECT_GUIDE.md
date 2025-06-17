# Guia do Projeto DevAI

Este arquivo permite descrever comandos e convenções específicas do projeto. O agente utiliza estas informações para compilar e testar corretamente.

## Comandos principais

- **Instalar dependências:** `pip install -r requirements.txt`
- **Executar testes:** `pytest`
- **Verificar estilo:** `flake8 devai`
- **Análise estática:** `pylint devai && mypy devai`
- **Análise de segurança:** `bandit -r devai`

Todos esses passos podem ser executados em sequência utilizando o script `./scripts/dev_checks.sh`.

Antes de enviar contribuições, recomendamos rodar `pre-commit run --files <arquivos>` localmente para garantir que formatação, lint, tipos e testes estejam corretos.

Adicione aqui quaisquer passos extras para build ou uso de frameworks (por exemplo, `npm install`, `mvn test`).
