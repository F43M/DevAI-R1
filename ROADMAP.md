# Roadmap

Este arquivo lista sugestões de melhorias para o DevAI. Sinta-se livre para contribuir adicionando novas ideias.

Consulte tambem `performance_roadmap.md` e `UI_roadmap.md` para direcoes especificas de desempenho e interface.

## Melhorias identificadas

- **Expansão dos módulos**: implementar novas funcionalidades nos módulos existentes e criar outros conforme necessário.
- **Cobertura de testes**: incluir testes para `tasks.py`, `core.py`, `cli.py` e demais componentes.
- **Dependências opcionais**: oferecer fallback ou mensagens de erro mais amigáveis caso `sentence_transformers` ou `faiss` não estejam instalados (implementado; ver `UI_roadmap.md`).
- **Exemplos de configuração**: disponibilizar modelos de `config.yaml` e `tasks.yaml` para facilitar o uso.
- (adicione novos itens aqui)
- **Persistência de histórico via localStorage**
- **Persistência de histórico de sessões entre reinicializações**
- **Sistema simbólico de carregamento para ações demoradas**
- **Ajuda simbólica para desenvolvedores iniciantes (GUI)**
- **Onboarding simbólico com fluxo passo a passo**
- **Agrupamento simbólico de funções para reduzir carga cognitiva inicial**
- **Melhoria simbólica** – Validação defensiva aplicada em todas as rotas e edições de arquivo.
- **Raciocínio progressivo** com modo `step_by_step` configurável no `project_identity.yaml`.
- **Verificação simbólica de coerência** registrando pontuação e detalhes no `prompt_log.jsonl`.
- **Tagging automático de memória** para funções novas ou refatoradas (`symbolic_memory_tagger.py`).
- **AutoReview** periódico detectando código sem docstring ou não utilizado.
- **reason_stack** registrando decisões em tarefas longas.
- **Proteção de código** via blocos `# <protect>` ... `# </protect>`.
- **decision_log.yaml** para auditoria de cada ação da IA.
- **Validação simbólica e notificação de API key**
- **Modo observador** ativado com `--observer` para apenas registrar insights.
- **Prompt de metacognição** sugerindo próximos passos a partir do histórico.
- **Versionamento de respostas** em `history/prompts/` com hash e diff.
- **Automação incremental**: permitir que o projeto evolua com novas funções de forma contínua.
- **Cache de memória**: reutilizar embeddings e consultas frequentes para acelerar o aprendizado.
- **Memória simbólica a partir de sessões**
- **Relatórios de cobertura**: integrar ferramentas como `coverage.py` ao processo de testes.
- **Testes de integração simbólica** validando o ciclo completo IA → sugestão → teste → aplicação.
- **Análise de segurança**: incorporar `bandit` para detectar vulnerabilidades.
- **Integração contínua**: rodar tarefas de teste e análise estática automaticamente em um pipeline CI.
- **Métricas avançadas**: monitorar uso de CPU e memória do assistente.
- **Sistema de plugins** para extensões de tarefas. *(implementado)*
- **Refatoração dos plugins**: simplificar registro e testes das extensões.
- **Suporte a múltiplos backends de IA**.
- **Notificações automáticas** ao concluir tarefas longas.
- **Refatoração automática validada por testes** *(implementado)*
- **Monitoramento de complexidade do código ao longo do tempo** *(implementado)*
- **Integração com IDEs (VSCode, etc.)** *(futuro)*
- **Treinamento incremental com dados do histórico** *(futuro)*
- **Fine-tuning com RLHF** *(experimental)* – use `python -m devai.rlhf <modelo> <saida>`
- **Avaliar modularização das dependências pesadas** (TRL, Transformers, Textual)
  movendo `rlhf.py`, `ui.py` e `tui.py` para `plugins/` ou repositório separado,
  reduzindo o peso da instalação padrão.
- **Sandbox de execução com containers** *(implementado)*
- **Prompts com Chain-of-Thought** *(parcialmente implementado)*
- **Planejamento multi-turn interativo** *(futuro)*
- **Confirmação antes de ações críticas** *(futuro)*
- **Painel web estilo IDE** com diffs e chat integrado *(futuro)*
- **Componente de metacognição** para aprendizado contínuo *(futuro)*
- **Paralelismo de tarefas** acelerando lint e testes *(implementado em parte)*
- **Uso de modelos menores** para otimizar embeddings *(futuro)*
- **Padronização simbólica de termos e responsividade total**
- **Renderização semântica para resultados complexos**


## Melhoria pendente – Função /analyze
- Implementar contexto multi-turno para /analyze via self.conversation_history persistente.

## Melhoria pendente – Análise Explicada (/analyze_deep)
- Avaliar estabilidade da divisão plano/resposta e ajustar a UI conforme necessário.

## Melhoria pendente – Análise de Projeto
- DONE: Aprimorar visualização colorida do relatório gerado (ver `UI_roadmap.md`).
- Permitir execução assíncrona do deep_scan_app. (ver `performance_roadmap.md`)
- Controle de modo de inicialização do DevAI

## Melhoria pendente – Aprendizado Simbólico
- Implementar rastreamento de origem das regras aprendidas
- Executar treinamento simbólico de forma assíncrona
- Autoavaliação com rastreamento simbólico por regra

## Melhoria implementada – Aplicação de refatoração simulada
- Código do /dry_run pode ser aplicado automaticamente com backup e rollback.

## Melhoria implementada – Análise inteligente do test_output
- Falhas individuais são destacadas e resumos mais claros são gerados.

## Melhoria pendente – Encerramento completo de recursos dinâmicos
- Encerrar watchers e ciclos longos de forma previsível no método `shutdown`.

## Melhoria implementada – Sandbox de execuções de teste
- Sandbox de execução com limite de CPU e memória para testes isolados.

## Adaptive Prompt Construction
- Implementar filtragem condicional de contextos no prompt.
- Registrar no log quais blocos foram incluídos e por quê.
- Ajustar a frase "Explique antes de responder." apenas quando necessário.
- #future-enhancement:intent-routing

## MULTI_TURN_CHAT
- Persistir histórico de conversa por sessão via `ConversationHandler`.
- Limitar o contexto às 10 mensagens mais recentes.
- Endpoint `/reset_conversation` aceita `session_id` e limpa apenas aquela sessão.
- Sincronizar automaticamente `chatHistory` local com o backend quando o endpoint `/history` estiver disponível.
