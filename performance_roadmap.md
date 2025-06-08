# Performance Roadmap

- # FUTURE: async scan
  Implementar execução assíncrona do `deep_scan_app` e armazenamento de status em segundo plano para grandes projetos.
- # FUTURE: run symbolic_training async
  Avaliar transformação do `/symbolic_training` em tarefa de segundo plano para não bloquear a interface.
- # FUTURE: limitar ciclos por etapa no auto_monitor_cycle
  Garantir que a autoavaliação não cause travamentos.
- Execução sob demanda do `deep_scan_app` configurada via `START_MODE`.
