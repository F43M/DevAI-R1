# DevAI-R1

Assistente de desenvolvimento baseado em IA com suporte a contextos de até **160k tokens** via OpenRouter. Agora inclui validação de configuração e autenticação com tokens JWT.

## Principais recursos

- **Memória persistente** com busca vetorial (FAISS)
- **Níveis de contexto** para memória de curto, médio e longo prazo
- **Cache de embeddings** para acelerar consultas repetidas
- **Limpeza automática de memórias** e feedback de uso
- **Análise de código** com construção de grafo de dependências
- **Parser multilíngue** (Python, JS, C++, HTML)
- **Métricas de complexidade** das funções analisadas
- **Acompanhamento da complexidade média do projeto**
- **Monitoramento de logs** detectando padrões de erro
- **Execução de testes automatizados e análise estática**
- **Relatórios de cobertura de testes**
- **Tarefas extras** com pylint e mypy
- **Análise de segurança com Bandit**
- **Monitoramento de complexidade ao longo do tempo**
- **Cache inteligente de prompts** reaproveitando respostas similares
- **Integração opcional com modelo local** para geração offline
- **API FastAPI** e **interface de linha de comando**
- **Interface web opcional** em `/static/index.html` para conversar com a IA e explorar arquivos
- Métricas expostas em `/metrics` (CPU e memória)
- **Histórico de uso de CPU/memória**
- **Histórico de tarefas** e sistema de plugins
- **Suporte a múltiplos modelos** configuráveis
- **Notificações por e-mail** opcionais
- **Integração contínua via GitHub Actions**
- **Refatoração automática validada por testes**
- **Prompts Chain-of-Thought** para melhor raciocínio
- **Estrutura inicial para fine-tuning via RLHF e sandbox de execução**

## Configuração

1. Utilize `config.example.yaml` e `tasks.example.yaml` como base. Copie-os para `config.yaml` e `tasks.yaml` e ajuste conforme necessário:

```yaml
CODE_ROOT: ./app
API_PORT: 8000
API_SECRET: "sua-chave"
```

2. Defina a variável de ambiente `OPENROUTER_API_KEY` com sua chave de acesso.

3. Instale as dependências do projeto para habilitar a comunicação real com o OpenRouter:

```bash
pip install -r requirements.txt
```

O DevAI traz versões simplificadas de algumas bibliotecas (como `aiohttp` e `fastapi`) usadas apenas em testes offline. O módulo `dependency_check` avisará caso essas versões estejam ativas, recomendando a instalação dos pacotes reais.

## Executando

- **Servidor API**:

```bash
python -m devai --api
```

Com o servidor ativo, acesse `http://localhost:8000/static/index.html` para utilizar a interface web.

- **Interface de linha de comando**:

```bash
python -m devai --cli
```

Os comandos disponíveis na CLI são listados ao iniciar o programa, como `/memoria`, `/tarefa` e `/grafo`. Para refatorar um arquivo automaticamente utilize:
`/tarefa auto_refactor caminho/para/arquivo.py`.

### Gerenciamento de arquivos

Além das tarefas padrão, a CLI permite explorar e modificar o diretório definido em `CODE_ROOT` (por padrão `./app`):

- `/ls [caminho]` lista arquivos e subpastas.
- `/abrir <arquivo> [ini] [fim]` exibe linhas específicas de um arquivo.
- `/editar <arquivo> <linha> <novo>` altera uma linha individual.
- `/tarefa auto_refactor <arquivo>` refatora o arquivo informado e executa os testes.

## Testes

Instale as dependências de desenvolvimento e execute:

```bash
pytest
```

O projeto inclui um arquivo `pyproject.toml` para facilitar a instalação das dependências e um arquivo `.pre-commit-config.yaml` com linters automáticos. Após instalar o `pre-commit`, execute `pre-commit install` para habilitar as verificações antes de cada commit.

## Plugins

Coloque scripts Python em `plugins/` para adicionar novas tarefas ao sistema.
Cada plugin deve implementar uma função `register(task_manager)`.
Veja `plugins/todo_counter.py` como exemplo.

Você também pode rodar os testes e a análise estática pelo gerenciador de tarefas:

```bash
python -m devai --cli
/tarefa run_tests
/tarefa static_analysis
```

### Integração contínua

O repositório inclui um workflow em `.github/workflows/ci.yml` que executa lint, análise de segurança e testes a cada *pull request*. Basta habilitar o GitHub Actions para que as tarefas sejam rodadas automaticamente.

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
- `update_manager.py` – aplica mudancas com rollback caso os testes falhem
- Diretório `plugins/` – extensões opcionais de tarefas

Sinta‑se livre para expandir cada módulo conforme necessário.
## Roadmap

Acompanhe [ROADMAP.md](ROADMAP.md) para sugestões de melhorias e futuras implementações.
Veja também o histórico de versões em [RELEASE_NOTES.md](RELEASE_NOTES.md).

Melhorias em andamento:

- Expansão dos módulos existentes
- Cobertura de testes ampliada
- Dependências opcionais com fallback *(implementado)*
- Exemplos de configuração prontos para uso
- Automação incremental do projeto
- Cache de memória para acelerar consultas
- Sistema de plugins para novas tarefas *(implementado)*
- Prompts com raciocínio em etapas *(Chain-of-Thought)*
- Estrutura para treinamento via RLHF
- Sandbox de execução para testes isolados *(planejado)*
- Relatórios de cobertura integrados
- Monitoramento de complexidade ao longo do tempo
  (histórico salvo em `complexity_history.json`)
- (adicione novas ideias aqui)

Uma versão resumida deste documento está disponível em `README_en.md` para facilitar contribuições internacionais.
