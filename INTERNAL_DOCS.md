# Internal Symbolic Learning

Este documento descreve como o DevAI captura conhecimento interno e realiza autoconhecimento.

## Ciclo de aprendizado simbólico
1. `learning_engine.py` percorre o código e analisa logs.
2. Cada trecho gera memórias classificadas (explicações, riscos, boas práticas).
3. Padrões positivos são sintetizados a partir de refatorações aprovadas.
4. O método `reflect_on_internal_knowledge` resume as lições recentes.

## Diferenca entre Learning Engine e Metacognition
- **Learning Engine** aplica análises e armazena resultados.
- **Metacognition** revisa decisões passadas e ajusta pontuações de módulos.
- Quando repetidas falhas são detectadas, a metacognição grava uma `licao aprendida` sugerindo revisão.

As memórias ficam disponíveis para novas rodadas de sugestão de código, reforçando práticas bem-sucedidas e evitando erros recorrentes.

## Memória simbólica de diálogo

1. O histórico de conversa pode ser resumido periodicamente pelo `DialogSummarizer`.
2. Esse resumo gera memórias simbólicas classificadas com tags de preferência do usuário ou lições aprendidas.
3. As memórias são armazenadas via `MemoryManager` com `memory_type` `dialog_summary`.
4. Em prompts futuros, blocos relevantes de memória são incluídos automaticamente.
5. Se nenhuma regra regex gerar memórias, é realizada uma etapa leve de resumo via modelo
   (`ENABLE_AI_SUMMARY`). O retorno deve conter linhas no formato `#tag: resumo` que são
   armazenadas pelo `MemoryManager`.

## pending_features
- ~~memory_extraction_fallback~~ (implementado)
- embedding_fallback
- Sincronização automática entre backend e chatHistory local para sessões multi-turn.
- Em sessões curtas, o botão “🧠 Contexto Atual” pode não retornar memórias ainda.
- Reset parcial de sessões (limpar conversa, mas manter memórias preferenciais)

## pending_rlhf
- Integração com `trl` para RLHF ainda não implementada.

#pending_logic: run_symbolic_training - origem das regras ainda não rastreada
#pending_logic: fine_tune - RLHF precisa da lib trl (não instalada)

## pending_fixes
- stub_fallback:fastapi
- stub_fallback:uvicorn
- stub_fallback:aiohttp
- stub_fallback:aiofiles
- stub_fallback:networkx
- stub_fallback:scikit-learn
