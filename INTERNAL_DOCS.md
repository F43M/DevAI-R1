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
