# Guia de API

Endpoints principais para integração e painel IDE:
- `POST /analyze` – gera resposta do modelo.
- `GET /files` – lista arquivos.
- `GET /file` – obtém conteúdo de um arquivo.
- `GET /actions` – retorna histórico de decisões do DevAI.
- `GET /diff?file=<nome>` – mostra diferença da última alteração registrada.
- `GET /approval_request` – aguarda até que o servidor solicite uma confirmação e retorna `{ "message": "texto", "token": "id" }`.
- `POST /approval_request` – envia `{ "approved": true|false, "token": "id" }` para responder à solicitação pendente.

Se `NOTIFY_EMAIL` ou `NOTIFY_SLACK` estiverem configurados, cada pedido de aprovação gera um e-mail ou mensagem com links diretos:
`http://localhost:<porta>/approval_request?token=<id>&approved=true` ou `...&approved=false`.
Ao clicar, o endpoint `POST /approval_request` é acionado e a decisão é registrada.

Estes endpoints permitem integração futura com VSCode ou JetBrains.
