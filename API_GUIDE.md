# Guia de API

Endpoints principais para integração e painel IDE:
- `POST /analyze` – gera resposta do modelo.
- `GET /files` – lista arquivos.
- `GET /file` – obtém conteúdo de um arquivo.
- `GET /actions` – retorna histórico de decisões do DevAI.
- `GET /diff?file=<nome>` – mostra diferença da última alteração registrada.

Estes endpoints permitem integração futura com VSCode ou JetBrains.
