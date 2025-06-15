# Guia Rápido

Este guia resume os passos iniciais para rodar o DevAI-R1.

1. **Clone o repositório**
   ```bash
   git clone <repo>
   cd DevAI-R1
   ```
2. **Instale as dependências**
   Os utilitários de desenvolvimento (`flake8`, `pylint`, `mypy`, `bandit` e
   `pre-commit`) estão listados em `requirements-dev.txt`.
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
3. **Copie e ajuste a configuração**
   ```bash
   cp config.example.yaml config.yaml
   # edite conforme necessário
   ```
4. **Crie o arquivo `.env`**
   ```bash
   echo OPENROUTER_API_KEY=suachave > .env
   ```
5. **Execute as verificações básicas**
   ```bash
   pre-commit install
   # opcional: ./scripts/dev_checks.sh para rodar linters e testes
   pytest
   ```
