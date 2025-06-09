# Configuração do DevAI

Todos os parâmetros de modelo devem ser definidos apenas em `MODELS.default` dentro do `config.yaml`:

```yaml
MODELS:
  default:
    name: deepseek/deepseek-r1-0528:free
    api_key: ${OPENROUTER_API_KEY}
    url: https://openrouter.ai/api/v1/chat/completions
```

O campo `MODEL_NAME` está **obsoleto** e será ignorado futuramente. Utilize `config.model_name` para obter o nome ativo do modelo em código.


## Limites de acesso
O DevAI valida caminhos fornecidos nas rotas e funções internas, impedindo leituras ou escritas fora de `CODE_ROOT`. Linhas negativas ou além do tamanho do arquivo resultam em erro.

## Isolamento de testes

Os testes automatizados podem ser executados em ambiente isolado. Utilize os campos abaixo no `config.yaml`:

```yaml
TESTS_USE_ISOLATION: true  # desative para rodar direto no sistema
TEST_CPU_LIMIT: 1          # limite de CPU (segundos) ou --cpus no Docker
TEST_MEMORY_LIMIT_MB: 512  # memória máxima em MB
```

Quando `TESTS_USE_ISOLATION` for `true`, o DevAI executará `pytest` em um container Docker com os limites configurados.

## Histórico de conversa

O parâmetro `MAX_SESSION_TOKENS` controla a quantidade máxima de tokens mantidos no arquivo de histórico de cada sessão. Ao exceder esse limite, as mensagens mais antigas são removidas (pruning). Defina `0` para desabilitar a limpeza automática.

A classe `ConversationHandler` oferece o método `search_history(session_id, query)` que utiliza embeddings gravados em `memory.db` para localizar mensagens similares ao texto ou tag informados.

