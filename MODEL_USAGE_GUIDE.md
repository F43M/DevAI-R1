# Guia de Uso do Modelo DeepSeek R1

Este projeto utiliza o modelo DeepSeek R1 via OpenRouter. Todas as requisições incluem uma mensagem de sistema definindo o papel do agente:
"Você é um assistente especialista em desenvolvimento de software com foco em qualidade, segurança e aprendizado simbólico."

As instruções solicitam que o modelo descreva em 2–3 etapas a lógica antes de gerar código e sempre apresente justificativa textual. Caso algum recurso do modelo não esteja disponível, documente aqui.

Atualmente não é possível integrar recursos avançados de aprendizado online do modelo. Se houver falha nos testes, o sistema gera um novo prompt de depuração automaticamente.

## Comportamento do provedor

Alguns provedores podem ignorar `max_tokens` e `temperature` especificados na
configuração. Em testes com o OpenRouter, o serviço costuma devolver até **4096
tokens** por resposta. O DevAI envia `stream=True` para transmitir os tokens aos
poucos; caso o modelo não ofereça streaming, o texto completo é retornado de uma
única vez. Consulte [limitations.md](limitations.md) para detalhes e para
entender eventuais variações observadas no endpoint `/analyze_deep`.
