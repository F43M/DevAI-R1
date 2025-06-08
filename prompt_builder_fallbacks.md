# Fallbacks para build_dynamic_prompt

- Se a pergunta não indicar claramente o contexto necessário, o prompt é gerado com todos os blocos disponíveis.
- Um log é registrado com a mensagem:
  "Fallback: prompt completo não foi simplificado por falta de sinal contextual".
- #future-enhancement:intent-routing: detectar a intenção da pergunta para evitar este fallback.
- Streaming de resposta é simulado dividindo o texto completo em tokens, pois o modelo atual não suporta `stream=True`.

