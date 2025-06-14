#!/usr/bin/env bash
# Executa verificacoes de desenvolvimento padrao para o DevAI-R1.

set -euo pipefail

flake8 devai
pylint devai
mypy devai
bandit -r devai
pytest
