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

### Parâmetros do sandbox

Outros comandos isolados utilizam a mesma infraestrutura. O diretório atual é
montado em `/app` com rede desativada e o diretório de trabalho dentro do
container também é `/app`.

```yaml
SANDBOX_IMAGE: python:3.10-slim
SANDBOX_CPUS: "1"
SANDBOX_MEMORY: 512m
SANDBOX_NETWORK: none
# SANDBOX_ALLOWED_HOSTS:
#   - example.com
```

`SANDBOX_NETWORK` define o modo de rede passado ao Docker (`bridge`, `host`,
`none`, etc.). O valor padrão `none` desativa totalmente o acesso. Caso
`SANDBOX_ALLOWED_HOSTS` contenha uma lista de domínios, uma rede temporária é
criada e regras de saída permitem conexões apenas para esses hosts.

Em Linux e macOS o Docker é usado quando disponível. No Windows é necessário o
Docker Desktop; caso ausente, o DevAI tenta rodar via WSL ou executa os comandos
diretamente exibindo um alerta. Nesse caso os limites de CPU/memória e o
controle de rede não estarão ativos.

Para rodar o DevAI em um ambiente totalmente isolado é possível criar uma imagem
Docker personalizada contendo todas as dependências (Git, compiladores, etc.) e
definir o caminho dessa imagem em `SANDBOX_IMAGE`. Basta criar um `Dockerfile`
baseado na distribuição de sua preferência, instalar todos os pacotes
necessários e executar `docker build -t minha/imagem .`.

## Histórico de conversa

O parâmetro `MAX_SESSION_TOKENS` controla a quantidade máxima de tokens mantidos no arquivo de histórico de cada sessão. Ao exceder esse limite, as mensagens mais antigas são removidas (pruning). Defina `0` para desabilitar a limpeza automática.

A classe `ConversationHandler` oferece o método `search_history(session_id, query)` que utiliza embeddings gravados em `memory.db` para localizar mensagens similares ao texto ou tag informados. Os embeddings de mensagens descartadas também são removidos do banco, evitando acúmulo de vetores antigos.

O resumo periódico das conversas executado pelo `DialogSummarizer` agora é assíncrono. Métodos como `summarize_conversation()` e a rotina interna `_summarize_and_store()` do `ConversationHandler` são `async` e devem ser aguardados quando chamados diretamente.

## Classificador de intenções

O arquivo `intent_samples.json` contém exemplos de frases e suas respectivas intenções. Adicione novos pares para ensinar o DevAI a reconhecer outras solicitações.

Após atualizar esse arquivo, execute o comando `/train_intents` no CLI. O processo irá gerar `intent_model.pkl`, utilizado pelo roteador de intenções. Caso o modelo não exista, o DevAI continuará usando apenas o mapeamento por palavras‑chave.

## Log de erros

O DevAI mantém um histórico dos erros ocorridos em `ERROR_LOG_PATH`.
Após cada execução, o arquivo é truncado para no máximo `ERROR_LOG_MAX_LINES` linhas para evitar crescimento indefinido.

```yaml
ERROR_LOG_PATH: errors_log.jsonl
ERROR_LOG_MAX_LINES: 1000
```

## Notificações

Para receber avisos automáticos, defina um e-mail ou um webhook do Slack:

```yaml
NOTIFY_EMAIL: ''       # opcional
NOTIFY_SLACK: ''       # opcional
```

Quando configurados, ambos os canais serão utilizados.

## Modos de aprovação

Use `APPROVAL_MODE` para definir o nível de confirmação exigido:

```yaml
APPROVAL_MODE: suggest
```

Valores aceitos:

- `full_auto` – nenhuma confirmação de operações;
- `auto_edit` – confirma somente comandos de shell;
- `suggest` – confirma ações de escrita ou execução de shell.

Comandos classificados como `shell_safe` são aprovados automaticamente,
mesmo no modo `suggest`. Use essa categoria para operações de leitura, como
`ls` ou `cat`. Demais execuções devem ser marcadas como `shell` e exigem
confirmação dependendo do modo configurado.

### Regras de autoaprovação

O campo `AUTO_APPROVAL_RULES` permite ignorar ou forçar confirmações
com base em padrões de caminho. Cada item deve conter `action`, `path`
e `approve`:

```yaml
AUTO_APPROVAL_RULES:
  - action: edit
    path: "docs/**"
    approve: true
```

`path` usa sintaxe glob. Se `approve` for `true` a ação é aplicada
sem perguntar; `false` exige confirmação mesmo em modos automáticos.

O comando `/sugerir_regras` pode analisar o `decision_log.yaml` e
propor entradas para esta lista automaticamente.

## Estilo de diff

O DevAI pode mostrar patches lado a lado ou no formato tradicional. Defina
`DIFF_STYLE` no `config.yaml`:

```yaml
DIFF_STYLE: inline  # ou side_by_side
```

O valor `inline` exibe o diff como texto único, enquanto `side_by_side`
separa as linhas em duas colunas.

Quando o usuário solicita alterações de código, o DevAI orienta o modelo a
retornar apenas um patch de diff unificado dentro de um bloco markdown `diff`.
Essa convenção facilita a aplicação automática descrita abaixo.

## Aplicação automática de diffs

Respostas da IA contendo um bloco começando por `diff --git` são tratadas como
patches. O DevAI divide o diff por arquivo, apresenta-o de acordo com o
`DIFF_STYLE` configurado e pergunta se deve aplicar as mudanças (dependendo do
`APPROVAL_MODE`). Se aceito, cada arquivo é atualizado e os testes são executados
automaticamente.

### Formato e aprovação

Respostas voltadas a modificações devem trazer apenas um patch unificado iniciado por `diff --git` dentro de um bloco de código `diff`. O DevAI emprega o helper `split_diff_by_file` para dividir diffs com múltiplos arquivos e o `apply_patch_to_file` (baseado no módulo `unidiff`) para aplicá-los com validação de contexto.

O processo segue as regras de `APPROVAL_MODE` e `AUTO_APPROVAL_RULES`: o diff é mostrado conforme `DIFF_STYLE` e pode ser aplicado automaticamente ou mediante confirmação.

Exemplo simplificado:

```diff
diff --git a/app/exemplo.py b/app/exemplo.py
--- a/app/exemplo.py
+++ b/app/exemplo.py
@@
-print("oi")
+print("olá mundo")
```

Após aprovar, o arquivo é atualizado e os testes configurados são executados.
