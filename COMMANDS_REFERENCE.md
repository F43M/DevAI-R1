# DevAI Command Reference

Este documento descreve os comandos simbólicos disponíveis no DevAI via CLI. Todos eles produzem mensagens simbólicas e estão preparados para integração futura ao painel visual.

Ao iniciar a CLI com `python -m devai --cli` é apresentada uma interface colorida baseada em Rich. Um terminal simples pode usar `--plain`.

O comportamento de confirmação de ações é controlado pelo `APPROVAL_MODE`. Use
`--approval-mode` na linha de comando para sobrescrever o valor de `config.yaml`.

Exemplo:

```
┌─ DevAI ──────────────┐
│ >>> /rastrear app.py │
└──────────────────────┘
```

## /lembrar <conteúdo> [tipo:<tag>]
Armazena uma memória manualmente na base vetorial. Opcionalmente é possível definir uma tag de tipo.

Exemplo:
```bash
devai lembrar "Evitar variáveis globais" tipo:regra
```

## /esquecer <termo>
Desativa memórias relacionadas ao termo. Elas não são removidas fisicamente, apenas marcadas como desativadas.

## /ajustar estilo:<parâmetro> valor:<opção>
Altera preferências de comportamento da IA. As configurações ficam em `PREFERENCES_STORE.json`.

Exemplo:
```bash
devai ajustar estilo:modo_refatoracao valor:seguro
```

## /rastrear <arquivo|tarefa>
Exibe o histórico simbólico de decisões e modificações que afetaram o alvo informado.

## /decisoes [lembrar|esquecer <id>] [acao:<tipo>] [arquivo:<caminho>]
Mostra as últimas entradas de `decision_log.yaml` com filtros opcionais por tipo de ação ou arquivo e permite marcar ou desmarcar a opção de lembrar aprovações.

## /memoria tipo:<tag> [filtro:<texto>]
Busca memórias armazenadas filtrando por tipo e texto. Possui paginação e a flag `--detalhado` para mostrar informações completas.

## /historia [sessao]
Exibe o histórico completo de mensagens trocadas com a IA. Caso nenhum `sessao` seja informado, usa "default".

## /refatorar <arquivo>
Aciona a refatoração automática para o arquivo informado.

Exemplo:
```bash
devai refatorar src/modulo.py
```

## /rever <arquivo>
Executa uma revisão automática de código no arquivo indicado.

Exemplo:
```bash
devai rever src/modulo.py
```

## /resetar
Limpa o histórico de conversa da sessão atual.

Exemplo:
```bash
devai resetar
```

## /historico_cli [N]
Exibe o log completo da CLI. Informe `N` para limitar às últimas linhas.

Exemplo:
```bash
devai historico_cli 50
```

## /modo <suggest|auto_edit|full_auto>
Define rapidamente o `APPROVAL_MODE` sem editar arquivos.

Exemplo:
```bash
devai modo auto_edit
```

## /aprovar_proxima [N]
Ativa aprovações automáticas temporárias para as próximas `N` ações.
O padrão é `1`.

Exemplo:
```bash
devai aprovar_proxima 3
```

## /regras [add <acao> <caminho> <sim|nao>|del <id>]
Gerencia as `AUTO_APPROVAL_RULES` do `config.yaml`. Sem argumentos apenas lista
as regras atuais numeradas.

Exemplo:
```bash
devai regras
devai regras add edit docs/** sim
devai regras del 1
```

## /ajuda
Mostra esta referência de comandos diretamente na CLI.

Exemplo:
```bash
devai ajuda
```

### Modos de aprovação

Defina `APPROVAL_MODE` em `config.yaml` ou via `--approval-mode`:

- `full_auto` aplica tudo automaticamente;
- `auto_edit` confirma apenas comandos de shell;
- `suggest` confirma alterações de código e comandos externos.
Também é possível alternar dinâmicamente usando `/modo`.

Para exceções específicas utilize `AUTO_APPROVAL_RULES` no `config.yaml`:

```yaml
AUTO_APPROVAL_RULES:
  - action: edit
    path: "docs/**"
    approve: true
```

O campo `path` aceita padrões glob.
