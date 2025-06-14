# Contribuindo para o DevAI-R1

Obrigado por querer contribuir! Siga estas etapas para propor melhorias:

1. Crie um fork do repositório e abra sua branch de trabalho.
2. Instale as dependências de desenvolvimento e execute `pre-commit install`.
3. Rode `pre-commit run --all-files` para aplicar `black`, `flake8`, `pylint` e `bandit` automaticamente.
4. Execute `pytest` para garantir que a suíte continua verde.
5. Envie um pull request descrevendo suas mudanças.

## Estilo de código
- Siga a convenção PEP8.
- Utilize `black` para formatação e `isort` para organizar imports (já incluídos no pre-commit).

## Como rodar a suíte de testes
- Testes unitários podem ser executados com:
  ```bash
  pytest
  ```
- Também é possível iniciar a CLI e usar o gerenciador de tarefas:
  ```bash
  python -m devai --cli
  /tarefa run_tests
  ```

## Verificações de desenvolvimento

Execute todos os linters e testes de uma vez com o script abaixo:

```bash
./scripts/dev_checks.sh
```

Feedback e sugestões são sempre bem-vindos.
