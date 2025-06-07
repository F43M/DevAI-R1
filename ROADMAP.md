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

