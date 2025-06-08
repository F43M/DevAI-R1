# Guia de UX

O painel web do DevAI inclui dicas rápidas para auxiliar quem está começando a utilizar a ferramenta.

- **Tooltips**: ao passar o mouse sobre cada botão principal, um texto curto explica o que aquela ação faz.
- **Botão "Ajuda"**: fica fixo na tela e abre um pequeno guia contextual detalhando o uso de cada função.
- **Comportamento adaptativo**: se novas dicas forem adicionadas, elas aparecerão automaticamente no overlay.

As ações mais complexas ficam agrupadas no menu **⚙️ Funções Avançadas**, deixando visíveis apenas os botões de análise simples e de projeto. Essa separação reduz a carga cognitiva para quem está começando.

Um painel inicial de boas-vindas explica os três passos básicos: abrir um arquivo, fazer uma pergunta e conferir a resposta. Ele some automaticamente assim que um arquivo é carregado. O mesmo passo a passo aparece na primeira execução via CLI, com sugestão para usar `--guided` se desejar mais explicações.

As dicas estão disponíveis apenas na interface web. Em execução headless, consulte `ui_fallbacks.md`.
