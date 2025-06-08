# Roadmap

Este arquivo lista sugestões de melhorias para o DevAI. Sinta-se livre para contribuir adicionando novas ideias.

## Melhorias identificadas

- **Expansão dos módulos**: implementar novas funcionalidades nos módulos existentes e criar outros conforme necessário.
- **Cobertura de testes**: incluir testes para `tasks.py`, `core.py`, `cli.py` e demais componentes.
- **Dependências opcionais**: oferecer fallback ou mensagens de erro mais amigáveis caso `sentence_transformers` ou `faiss` não estejam instalados. *(implementado)*
- **Exemplos de configuração**: disponibilizar modelos de `config.yaml` e `tasks.yaml` para facilitar o uso.
- (adicione novos itens aqui)
- **Raciocínio progressivo** com modo `step_by_step` configurável no `project_identity.yaml`.
- **Verificação simbólica de coerência** registrando pontuação e detalhes no `prompt_log.jsonl`.
- **Tagging automático de memória** para funções novas ou refatoradas (`symbolic_memory_tagger.py`).
- **AutoReview** periódico detectando código sem docstring ou não utilizado.
- **reason_stack** registrando decisões em tarefas longas.
- **Proteção de código** via blocos `# <protect>` ... `# </protect>`.
- **decision_log.yaml** para auditoria de cada ação da IA.
- **Modo observador** ativado com `--observer` para apenas registrar insights.
- **Prompt de metacognição** sugerindo próximos passos a partir do histórico.
- **Versionamento de respostas** em `history/prompts/` com hash e diff.
- **Automação incremental**: permitir que o projeto evolua com novas funções de forma contínua.
- **Cache de memória**: reutilizar embeddings e consultas frequentes para acelerar o aprendizado.
- **Relatórios de cobertura**: integrar ferramentas como `coverage.py` ao processo de testes.
- **Análise de segurança**: incorporar `bandit` para detectar vulnerabilidades.
- **Integração contínua**: rodar tarefas de teste e análise estática automaticamente em um pipeline CI.
- **Métricas avançadas**: monitorar uso de CPU e memória do assistente.
- **Sistema de plugins** para extensões de tarefas. *(implementado)*
- **Suporte a múltiplos backends de IA**.
- **Notificações automáticas** ao concluir tarefas longas.
- **Refatoração automática validada por testes** *(implementado)*
- **Monitoramento de complexidade do código ao longo do tempo** *(implementado)*
- **Integração com IDEs (VSCode, etc.)** *(futuro)*
- **Treinamento incremental com dados do histórico** *(futuro)*
- **Fine-tuning com RLHF** *(planejado)*
- **Sandbox de execução com containers** *(planejado)*
- **Prompts com Chain-of-Thought** *(parcialmente implementado)*
- **Planejamento multi-turn interativo** *(futuro)*
- **Confirmação antes de ações críticas** *(futuro)*
- **Painel web estilo IDE** com diffs e chat integrado *(futuro)*
- **Componente de metacognição** para aprendizado contínuo *(futuro)*
- **Paralelismo de tarefas** acelerando lint e testes *(implementado em parte)*
- **Uso de modelos menores** para otimizar embeddings *(futuro)*


## Melhoria pendente – Função /analyze
- Implementar contexto multi-turno para /analyze via self.conversation_history persistente.

## Melhoria pendente – Análise Explicada (/analyze_deep)
- Avaliar estabilidade da divisão plano/resposta e ajustar a UI conforme necessário.
\n## Melhoria pendente – Análise de Projeto\n- Aprimorar visualização colorida do relatório gerado.\n- Permitir execução assíncrona do deep_scan_app.
- Controle de modo de inicialização do DevAI

## Melhoria pendente – Aprendizado Simbólico
- Implementar rastreamento de origem das regras aprendidas
- Executar treinamento simbólico de forma assíncrona
- Autoavaliação com rastreamento simbólico por regra

## Melhoria pendente – Aplicação de refatoração simulada
- Permitir aplicar automaticamente o código sugerido pelo /dry_run preservando backup.

## Melhoria pendente – Análise inteligente do test_output
- Filtrar falhas individuais e exibir mensagens mais claras no resumo da simulação.

## Melhoria pendente – Encerramento completo de recursos dinâmicos
- Encerrar watchers e ciclos longos de forma previsível no método `shutdown`.

## Melhoria pendente – Sandbox de execuções de teste
- Aplicar sandbox de execução com limite de CPU e memória para testes isolados.
