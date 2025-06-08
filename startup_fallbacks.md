# Fallbacks de Inicialização

- `deep_scan_app` e `watch_app_directory` são executados apenas quando `START_MODE` é `full`.
- Em modo `fast`, o backend registra que a varredura ficou para execução sob demanda.
- O modo `custom` permite definir em `START_TASKS` quais tarefas (scan, watch, monitor) serão iniciadas.
