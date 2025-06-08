# Guia de Testes

Este documento explica como executar e o que validar nos testes.

- `pytest` roda todos os testes unitários e simbólicos.
- Os testes de integração simulam o ciclo IA → sugestão → teste → aplicação.
- Rode-os sempre que alterar fluxos principais do DevAI.

Alguns testes estão marcados com `@pytest.mark.skip` pois a memória multi-turno
não está habilitada. Consulte `testing_fallbacks.md` para detalhes.
