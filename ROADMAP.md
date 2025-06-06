# Roadmap

Este arquivo lista sugestões de melhorias para o DevAI. Sinta-se livre para contribuir adicionando novas ideias.

## Melhorias identificadas

- **Expansão dos módulos**: implementar novas funcionalidades nos módulos existentes e criar outros conforme necessário.
- **Cobertura de testes**: incluir testes para `tasks.py`, `core.py`, `cli.py` e demais componentes.
- **Dependências opcionais**: oferecer fallback ou mensagens de erro mais amigáveis caso `sentence_transformers` ou `faiss` não estejam instalados. *(implementado)*
- **Exemplos de configuração**: disponibilizar modelos de `config.yaml` e `tasks.yaml` para facilitar o uso.
- (adicione novos itens aqui)
- **Automação incremental**: permitir que o projeto evolua com novas funções de forma contínua.
- **Cache de memória**: reutilizar embeddings e consultas frequentes para acelerar o aprendizado.
- **Relatórios de cobertura**: integrar ferramentas como `coverage.py` ao processo de testes.
- **Análise de segurança**: incorporar `bandit` para detectar vulnerabilidades.
- **Integração contínua**: rodar tarefas de teste e análise estática automaticamente em um pipeline CI.
- **Métricas avançadas**: monitorar uso de CPU e memória do assistente.

