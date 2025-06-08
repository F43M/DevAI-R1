# Guia de UX

O painel web do DevAI inclui dicas rápidas para auxiliar quem está começando a utilizar a ferramenta.

- **Tooltips**: ao passar o mouse sobre cada botão principal, um texto curto explica o que aquela ação faz.
- **Botão "Ajuda"**: fica fixo na tela e abre um pequeno guia contextual detalhando o uso de cada função.
- **Comportamento adaptativo**: se novas dicas forem adicionadas, elas aparecerão automaticamente no overlay.

As ações mais complexas ficam agrupadas no menu **⚙️ Funções Avançadas**, deixando visíveis apenas os botões de análise simples e de projeto. Essa separação reduz a carga cognitiva para quem está começando.

Um painel inicial de boas-vindas explica os três passos básicos: abrir um arquivo, fazer uma pergunta e conferir a resposta. Ele some automaticamente assim que um arquivo é carregado. O mesmo passo a passo aparece na primeira execução via CLI, com sugestão para usar `--guided` se desejar mais explicações.

As dicas estão disponíveis apenas na interface web. Em execução headless, consulte `ui_fallbacks.md`.

## Nomenclatura simbólica

Os botões do painel utilizam termos curtos e fáceis de entender.
Exemplos:

- 📊 **Analisar Projeto** – substitui o antigo "Deep Analysis".
- 🧠 **Aprendizado com Erros** – substitui "Treinar com base em erros".
- 🧭 **Monitoramento Automático** – substitui "Auto Monitor".

Todas essas strings ficam definidas em `static/lang.js` para permitir traduções futuras e consistência entre o código e a interface.

## Indicadores de carregamento

Quando uma ação pode levar mais tempo (como aprendizado simbólico ou monitoramento automático), o painel exibe o texto "Processando..." com um pequeno efeito de piscar. Todos os botões são desativados até a conclusão, evitando cliques repetidos.
Desde esta versão, as respostas da IA são enviadas em tempo real pelo endpoint `/analyze_stream`. O console vai sendo preenchido token a token, oferecendo feedback imediato de que o sistema está ativo.

## Exibição de resultados complexos

Resultados retornados em JSON agora são formatados em blocos coloridos no painel.
- **Verde** indica sucesso ou resumo.
- **Laranja** mostra sugestões de melhoria.
- **Vermelho** destaca riscos ou alertas.
Caso o formato não seja reconhecido, o JSON bruto é exibido com indentação para facilitar a leitura.

## Recuperação automática da sessão

O painel web salva o histórico recente de interações no `localStorage`.
Ao recarregar a página, o DevAI restaura o conteúdo do painel e do console,
mostrando a mensagem:
"🔄 Sessão recuperada – continue de onde parou.". Há também um botão
**🧹 Limpar Sessão** que apaga os dados salvos e confirma a ação no console.

## Contexto de Conversa

O DevAI mantém o histórico de cada sessão para oferecer respostas mais
coerentes em interações de múltiplos turnos. As últimas mensagens são
reenviadas ao modelo sempre que você faz uma nova pergunta. Use o botão
"🔄 Nova conversa" ou o endpoint `/reset_conversation` para limpar o
histórico atual quando desejar começar do zero.
Com a nova implementação de histórico estruturado, o assistente entende
referências a perguntas anteriores e mantém o raciocínio encadeado.

## Chave de API ausente ou inválida

Se a variável `OPENROUTER_API_KEY` não estiver configurada ou for rejeitada pelo
servidor, o painel exibirá:

```
🚫 Nenhuma chave de API foi detectada. Configure OPENROUTER_API_KEY para habilitar a IA.
```

Isso evita erros confusos e orienta o usuário a editar o `.env` ou `config.yaml`
com a chave correta.
