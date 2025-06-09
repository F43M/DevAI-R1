# DevAI Command Reference

Este documento descreve os comandos simbólicos disponíveis no DevAI via CLI. Todos eles produzem mensagens simbólicas e estão preparados para integração futura ao painel visual.

Ao iniciar a CLI com `python -m devai --cli` é apresentada uma interface colorida baseada em Rich. Um terminal simples pode usar `--plain`.

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
