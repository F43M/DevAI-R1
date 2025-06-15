# Guia de Testes

Este documento explica como executar e o que validar nos testes.

- `pytest` roda todos os testes unitários e simbólicos.
- Os testes de integração simulam o ciclo IA → sugestão → teste → aplicação.
- Rode-os sempre que alterar fluxos principais do DevAI.

Todos os testes são executados normalmente. O suporte a memória multi-turno está
ativo, portanto não há casos marcados com `skip`.

Para rodar a suíte sem acesso à internet, defina a variável
`HUGGINGFACE_HUB_OFFLINE=1` e baixe previamente os modelos utilizados pelo
`transformers`.
