# Guia de API

A aplicação expõe uma API FastAPI para interação com a IA e operações de gerenciamento de arquivos. Abaixo estão todos os endpoints disponíveis com exemplos de chamada usando `curl`.

## Conversa e memória
- `POST /analyze` – gera resposta do modelo.
  ```bash
  curl -X POST http://localhost:8000/analyze -d "query=print('ola')"
  ```
- `GET /analyze_stream` – versão em streaming do endpoint anterior.
  ```bash
  curl "http://localhost:8000/analyze_stream?query=exemplo"
  ```
- `POST /reset_conversation` – reinicia o histórico de uma sessão.
  ```bash
  curl -X POST http://localhost:8000/reset_conversation -d "session_id=default"
  ```
- `POST /session/reset` – remove todos os dados de uma sessão.
- `GET /session/history` – recupera o histórico completo da sessão.
- `GET /history` – alias para o histórico da sessão padrão.
- `POST /analyze_deep` – análise detalhada retornando plano e resposta.
- `GET /memory` – busca memórias salvas (`query`, `top_k`, `level`).
- `POST /feedback` – registra feedback `{memory_id, is_positive}`.

## Arquivos
- `GET /files` – lista diretórios abaixo de `CODE_ROOT`.
  ```bash
  curl "http://localhost:8000/files?path=subpasta"
  ```
- `GET /file` – retorna linhas de um arquivo (`file`, `start`, `end`).
- `POST /file/edit` – edita linha específica (requer token).
- `POST /file/create` – cria novo arquivo (requer token).
- `POST /file/delete` – remove arquivo (requer token).
- `POST /dir/create` – cria diretório (requer token).
- `POST /dir/delete` – remove diretório (requer token).
- `GET /file_history` – histórico de modificações de um arquivo.
- `GET /diff` – diff da última alteração de um arquivo.

## Controle de refatoração
- `POST /dry_run` – executa simulação de patch e testes.
- `POST /apply_refactor` – aplica código sugerido (requer token).
- `GET /deep_analysis` – gera relatório completo do projeto.

## Monitoramento e tarefas
- `GET /status` – informações gerais e tarefas em execução.
- `GET /metrics` – métricas de CPU e memória.
- `GET /logs/recent` – últimos registros de log.
- `GET /admin` – status de indexação.
- `POST /admin/rescan` – força nova varredura de código.
- `GET /auto_monitor` – executa ciclo de monitoramento automático.
- `GET /monitor/history` – histórico das execuções do monitor.
- `GET /complexity/history` – evolução da complexidade média.
- `GET /metacognition/summary` – arquivos com baixa pontuação.
- `POST /memory/optimize` – compacta e remove memórias antigas.
- `POST /learning/lessons` – resume lições aprendidas.
- `POST /symbolic_training` – dispara treinamento simbólico (requer token).
- `POST /train/rlhf` – executa fine‑tuning RLHF (requer token).

## Aprovação
- `GET /approval_request` – aguarda solicitação de aprovação e retorna `{message, details, token}`.
- `POST /approval_request` – envia decisão `{approved, token}`.
- `GET /actions` – lista de decisões gravadas.
- `GET /session/context` – exibe contexto resumido da sessão.
- `GET /context/search` – procura mensagens com uma tag específica.

Todos os endpoints que exigem autenticação usam o token definido em `API_SECRET` via parâmetro `token`.
