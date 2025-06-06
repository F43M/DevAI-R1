# DevAI-R1

Assistente de desenvolvimento baseado em IA com suporte a contextos de até **160k tokens** via OpenRouter.

## Principais recursos

- **Memória persistente** com busca vetorial (FAISS)
- **Cache de embeddings** para acelerar consultas repetidas
- **Limpeza automática de memórias** e feedback de uso
- **Análise de código** com construção de grafo de dependências
- **Monitoramento de logs** detectando padrões de erro
- **Execução de testes automatizados e análise estática**
- **API FastAPI** e **interface de linha de comando**
- Métricas expostas em `/metrics`

## Configuração

1. Utilize `config.example.yaml` e `tasks.example.yaml` como base. Copie-os para `config.yaml` e `tasks.yaml` e ajuste conforme necessário:

```yaml
CODE_ROOT: ./app
API_PORT: 8000
```

2. Defina a variável de ambiente `OPENROUTER_API_KEY` com sua chave de acesso.

## Executando

- **Servidor API**:

```bash
python -m devai --api
```

- **Interface de linha de comando**:

```bash
python -m devai --cli
```

Os comandos disponíveis na CLI são listados ao iniciar o programa, como `/memoria`, `/tarefa` e `/grafo`.

### Gerenciamento de arquivos

Além das tarefas padrão, a CLI permite explorar e modificar o diretório definido em `CODE_ROOT` (por padrão `./app`):

- `/ls [caminho]` lista arquivos e subpastas.
- `/abrir <arquivo> [ini] [fim]` exibe linhas específicas de um arquivo.
- `/editar <arquivo> <linha> <novo>` altera uma linha individual.

## Testes

Instale as dependências de desenvolvimento e execute:

```bash
pytest
```

Você também pode rodar os testes e a análise estática pelo gerenciador de tarefas:

```bash
python -m devai --cli
/tarefa run_tests
/tarefa static_analysis
```

## Estrutura modular

O código foi dividido em módulos dentro do pacote `devai/`, facilitando a inclusão de novas funcionalidades:

- `config.py` – carregamento de configuração e logging
- `memory.py` – gerenciamento de memórias
- `analyzer.py` – análise de código e grafo de dependências
- `tasks.py` – execução de tarefas
- `log_monitor.py` – monitoramento de arquivos de log
- `ai_model.py` – comunicação com o OpenRouter
- `core.py` – orquestração principal
- `cli.py` – interface de linha de comando
- `lint.py` – checagem simples de TODOs

Sinta‑se livre para expandir cada módulo conforme necessário.
## Roadmap

Acompanhe [ROADMAP.md](ROADMAP.md) para sugestões de melhorias e futuras implementações.
