# Fallbacks de Multi-Turn Chat

- Caso o `session_id` não seja informado ou a memória da sessão não possa ser recuperada, o DevAI responde em modo single-turn.
- Esse comportamento é registrado no log com a chave `multi_turn_fallback`.
