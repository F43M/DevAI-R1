# Guia de API

Endpoints principais para integração e painel IDE:
- `POST /analyze` – gera resposta do modelo.
- `GET /files` – lista arquivos.
- `GET /file` – obtém conteúdo de um arquivo.
- `GET /actions` – retorna histórico de decisões do DevAI.
- `GET /diff?file=<nome>` – mostra diferença da última alteração registrada.
- `GET /approval_request` – aguarda até que o servidor solicite uma confirmação e retorna `{ "message": "texto" }`.
- `POST /approval_request` – envia `{ "approved": true|false }` para responder à solicitação pendente.

Estes endpoints permitem integração futura com VSCode ou JetBrains.
