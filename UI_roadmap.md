# Planejamento de UI

A área de console agora exibe o plano de raciocínio retornado pelo endpoint
`/analyze_deep`. A separação visual depende do modelo seguir corretamente a
marcação `===RESPOSTA===`. Se falhas frequentes ocorrerem, essa divisão poderá
ser revertida até que a IA tenha comportamento estável.

- DONE: relatório do /deep_analysis agora colore linhas conforme severidade.
- DONE: marcação `===RESPOSTA===` interpretada mesmo com espaços extras.
- Se `rich` falhar ou não estiver instalado, a CLI automaticamente usa o modo `--plain`.
- Consulte `performance_roadmap.md` para metas de desempenho relacionadas.
