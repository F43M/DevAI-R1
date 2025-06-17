# DevAI-R1

Assistente de desenvolvimento baseado em IA com suporte a contextos de até **160k tokens** via OpenRouter. Agora inclui validação de configuração e autenticação com tokens JWT.

Para um guia rápido de configuração, veja [docs/QUICKSTART.md](docs/QUICKSTART.md).


## Principais recursos

- **Memória persistente** com busca vetorial (FAISS)
- **Níveis de contexto** para memória de curto, médio e longo prazo
- **Cache de embeddings** para acelerar consultas repetidas
- **Limpeza automática de memórias** e feedback de uso
- **Análise de código** com construção de grafo de dependências
- **Parser multilíngue** (Python, JS, C++, HTML)
- **Suporte expandido a linguagens** (Java, C#, Ruby, PHP)
- **Métricas de complexidade** das funções analisadas
- **Acompanhamento da complexidade média do projeto**
- **Monitoramento de logs** detectando padrões de erro
- **Execução de testes automatizados e análise estática**
- **Relatórios de cobertura de testes**
- **Tarefas extras** com pylint e mypy
- **Análise de segurança com Bandit**
- **Tarefa "Quality Suite" paralela para lint e testes**
- **Monitoramento de complexidade ao longo do tempo**
- **Cache inteligente de prompts** reaproveitando respostas similares
- **Integração opcional com modelo local** para geração offline
- **API FastAPI** e **interface de linha de comando**
- **Interface web opcional** em `/static/index.html` para conversar com a IA e explorar arquivos
- Métricas expostas em `/metrics` (CPU e memória)
- **Histórico de uso de CPU/memória**
- **Histórico de tarefas** e sistema de plugins
- **Plugin de contexto de frameworks** lendo `package.json`, `requirements.txt` ou `pom.xml`
- **Suporte a múltiplos modelos** configuráveis
- **Notificações por e-mail** opcionais
- **Integração contínua via GitHub Actions**
- **Refatoração automática validada por testes**
- **Prompts Chain-of-Thought** para melhor raciocínio
- **Suporte experimental a fine-tuning via RLHF**

## Configuração

1. Utilize `config.example.yaml` e `tasks.example.yaml` como base. Copie-os para `config.yaml` e `tasks.yaml` e ajuste conforme necessário:

```yaml
CODE_ROOT: ./app
API_PORT: 8000
API_SECRET: "sua-chave"
```

2. Crie um arquivo `.env` com `OPENROUTER_API_KEY=<sua chave>` ou defina essa
   variável diretamente no ambiente. O DevAI carrega esse arquivo
   automaticamente se o pacote `python-dotenv` estiver instalado.

3. Instale as dependências do projeto para habilitar a comunicação real com o OpenRouter e as ferramentas de desenvolvimento (linters, testes etc.):

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
# dependências extras para UI e RLHF (opcional)
pip install transformers trl rich prompt_toolkit textual
```

Caso pretenda utilizar a interface web incluída em `static/`, instale também as
dependências JavaScript:

```bash
npm install --prefix static
```

O índice vetorial utilizado na busca é salvo automaticamente em `faiss.index` e
`faiss_ids.json`. Mantenha esses arquivos para preservar o histórico de
consultas entre execuções. Caminhos customizados podem ser definidos no
`config.yaml` usando `INDEX_FILE` e `INDEX_IDS_FILE`.

4. Consulte `PROJECT_GUIDE.md` para uma lista de comandos extras, incluindo
   instruções de build para a UI e dicas de testes isolados via Docker.

Para executar a suíte de testes em ambiente isolado, configure no `config.yaml`:

```yaml
TESTS_USE_ISOLATION: true
TEST_CPU_LIMIT: 1
TEST_MEMORY_LIMIT_MB: 512
```

O parâmetro `APPROVAL_MODE` define quando o DevAI solicita confirmação antes de
executar ações sensíveis. Valores possíveis:

- `full_auto` – nenhuma confirmação é pedida;
- `auto_edit` – confirma apenas comandos de shell;
- `suggest` – confirma alterações de código e comandos externos.

O parâmetro `DIFF_STYLE` controla como os patches são exibidos na interface. Use
`inline` para o formato tradicional ou `side_by_side` para mostrar as mudanças em
duas colunas.

Quando a IA responde com um bloco `diff --git`, o DevAI divide o patch por
arquivo, exibe-o conforme o `DIFF_STYLE` escolhido e, se o `APPROVAL_MODE`
permitir, aplica automaticamente as alterações antes de rodar os testes
configurados.

Você pode ajustar no `config.yaml`, via `--approval-mode` ao iniciar
ou dinamicamente com o comando `/modo`.

Para manipular regras específicas de autoaprovação utilize `/regras`. O comando
permite listar, adicionar e remover entradas em `AUTO_APPROVAL_RULES` sem editar
o arquivo manualmente.

O módulo `dependency_check` alerta caso alguma dependência principal esteja ausente.

### Respostas em diff e aplicação de patches

Para alterações de código o modelo deve responder com um bloco `diff` iniciando por `diff --git`. O DevAI utiliza o novo helper `split_diff_by_file` para quebrar cada arquivo do patch e aplica as mudanças com `apply_patch_to_file`, que por sua vez depende do pequeno parser `unidiff` para validar o contexto.

O fluxo de aprovação respeita o `APPROVAL_MODE` e as regras em `AUTO_APPROVAL_RULES`. Caso o patch não esteja pré-aprovado, a interface mostra o diff formatado conforme `DIFF_STYLE` e solicita confirmação antes de rodar os testes definidos. Exemplo de resposta esperada:

```diff
diff --git a/app/exemplo.py b/app/exemplo.py
--- a/app/exemplo.py
+++ b/app/exemplo.py
@@
-print("oi")
+print("olá mundo")
```

Após a aprovação, o arquivo é atualizado e os testes são executados automaticamente.

## Windows

O sandbox de testes depende do Docker para isolar os processos. No Windows é
possível utilizar o [Docker Desktop](https://docs.docker.com/desktop/) ou
habilitar o suporte a containers no
[WSL2](https://learn.microsoft.com/windows/wsl/install). Após instalar uma das
opções, reinicie o terminal e confirme que o comando `docker` está disponível.
Sem o Docker, os comandos de shell são executados diretamente, sem isolamento.

## Executando

- **Servidor API**:

```bash
python -m devai --api
```

Com o servidor ativo, acesse `http://localhost:8000/static/index.html` para utilizar a interface web.

- **Interface de linha de comando (CLI)**:

```bash
python -m devai --cli
```
Adicione `--tui` para abrir a interface textual (TUI). Ambas utilizam o mesmo roteador de comandos em `devai/command_router.py`.

Por padrão a CLI utiliza a interface colorida do [Rich](https://github.com/Textualize/rich). Para um terminal mais simples use a flag `--plain`.

Atalhos comuns:
- setas **↑/↓** percorrem o histórico de comandos;
- **Tab** sugere comandos e caminhos de arquivo.

Exemplo de tela:

```
┌─ DevAI ───────────────────────┐
│ >>> /memoria                 │
│ Resultado destacado em cores  │
└───────────────────────────────┘
```

Os comandos disponíveis são listados ao iniciar o programa, como `/memoria`, `/tarefa` e `/grafo`. Use `/ajuda` para ver a descrição detalhada de cada um. Para refatorar um arquivo automaticamente utilize:
`/tarefa auto_refactor caminho/para/arquivo.py`.

- **Interface TUI (Textual)**:

```bash
python -m devai --cli --tui
```
Utiliza o mesmo conjunto de comandos da CLI, exibindo painéis de histórico e diff.

Os mesmos atalhos funcionam:

- setas **↑/↓** percorrem o histórico de comandos;
- **Tab** sugere comandos e caminhos de arquivo.

### Gerenciamento de arquivos

Além das tarefas padrão, a CLI permite explorar e modificar o diretório definido em `CODE_ROOT` (por padrão `./app`):

- `/ls [caminho]` lista arquivos e subpastas.
- `/abrir <arquivo> [ini] [fim]` exibe linhas específicas de um arquivo.
- `/editar <arquivo> <linha> <novo>` altera uma linha individual.
- `/novoarq <arquivo> [conteudo]` cria um novo arquivo.
- `/novapasta <caminho>` cria uma pasta.
- `/deletar <caminho>` remove arquivo ou diretório (requer confirmação).
- `/tarefa auto_refactor <arquivo>` refatora o arquivo informado e executa os testes.
- `/historia [sessao]` exibe o histórico de conversa da sessão indicada.
- `/historico_cli [N]` mostra N últimas linhas do log da CLI (ou todo o arquivo). 
- `/decisoes` lista as entradas do `decision_log.yaml`, indicando validade das
  decisões lembradas. Use `/decisoes purge` para remover registros expirados.
- `/modo <suggest|auto_edit|full_auto>` altera o nível de aprovação em tempo real.
- `/ajuda` exibe a documentação completa de comandos.

Ao usar `/deletar`, a CLI exibe um diálogo de confirmação para evitar remoções acidentais.
Caso escolha lembrar uma decisão, será solicitado por quantos dias ela deve permanecer válida.

### Histórico de complexidade

Com o servidor API rodando é possível consultar `/complexity/history` para obter
o histórico de variações de complexidade salvas em `complexity_history.json` e
a evolução da tendência gravada em `complexity_trend.json`.
O endpoint retorna uma lista de registros, a tendência média recente e os dados
agregados de tendência:

```json
{
  "history": [{"timestamp": "2024-01-01T12:00:00", "average_complexity": 3.2}],
  "trend": -0.1,
  "trend_history": [{"timestamp": "2024-01-01T12:00:00", "trend": -0.1}]
}
```

Valores negativos de `trend` indicam redução da complexidade média do projeto.

## Testes

Instale as dependências de desenvolvimento (que incluem `flake8`, `pylint`,
`mypy`, `bandit` e `pre-commit`) com:

```bash
pip install -r requirements-dev.txt
```

Em seguida execute:

```bash
pytest
```

Se estiver sem acesso à internet, defina `HUGGINGFACE_HUB_OFFLINE=1` e garanta
que os modelos necessários já estejam em cache (use `transformers-cli` ou
`AutoModel.from_pretrained` com rede habilitada uma vez).

O projeto inclui um arquivo `pyproject.toml` para facilitar a instalação das dependências e um arquivo `.pre-commit-config.yaml` com linters automáticos. Para habilitar as verificações locais instale e configure o `pre-commit`:

```bash
pip install pre-commit  # já incluído em requirements-dev.txt
pre-commit install
# opcional: roda todos os checks de uma vez
pre-commit run --all-files
# ou execute o script auxiliar
./scripts/dev_checks.sh
```

## Plugins

Coloque scripts Python em `plugins/` para adicionar novas tarefas ao sistema.
Cada plugin deve implementar uma função `register(task_manager)` e, opcionalmente,
`unregister(task_manager)`.
Veja `plugins/todo_counter.py` como exemplo.

### Habilitando ou desabilitando plugins

1. Inicie a CLI com `python -m devai --cli`.
2. Execute `/plugins` para listar o status de cada plugin.
3. Use `/plugin <nome> on` para ativar o plugin desejado.
4. Use `/plugin <nome> off` para desativá‑lo.
5. O estado fica salvo em `plugins.sqlite` e é carregado automaticamente.

Você também pode rodar os testes e a análise estática pelo gerenciador de tarefas:

```bash
python -m devai --cli
/tarefa run_tests
/tarefa static_analysis
```

## Treinamento RLHF

Depois de registrar feedback positivo via API ou CLI, é possível refinar o modelo base utilizando a biblioteca [`trl`](https://github.com/huggingface/trl) e o `SFTTrainer`.
Instale `transformers` e `trl` via `pip` e execute:

```bash
python -m devai.rlhf <modelo_base> ./model_ft
# ou pela CLI interativa
/treinar_rlhf <modelo_base> [pasta_destino]
```

O comando coleta os exemplos do banco de memória, monta um pequeno dataset supervisionado e chama o `SFTTrainer` da `trl`. Os checkpoints do modelo e o arquivo `metrics.json` são gravados no diretório indicado.

O dataset combinado é salvo em `logs/rlhf_dataset.json` e um arquivo com o hash
SHA256 é criado em `logs/rlhf_results/datasets/`. O treinamento programado só é
executado quando esse hash muda.

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
Veja também o histórico de versões em [RELEASE_NOTES.md](RELEASE_NOTES.md) e o guia interno em [INTERNAL_DOCS.md](INTERNAL_DOCS.md).

Melhorias em andamento:

- Expansão dos módulos existentes
- Cobertura de testes ampliada
- Dependências opcionais
- Exemplos de configuração prontos para uso
- Automação incremental do projeto
- Cache de memória para acelerar consultas
- Sistema de plugins para novas tarefas *(implementado)*
- Prompts com raciocínio em etapas *(Chain-of-Thought)*
- Estrutura para treinamento via RLHF (execute `python -m devai.rlhf <modelo> <pasta>` ou use `/treinar_rlhf` pela CLI)
- Sandbox de execução para testes isolados *(implementado)*
- Relatórios de cobertura integrados
- Monitoramento de complexidade ao longo do tempo
  (histórico salvo em `complexity_history.json`)
- (adicione novas ideias aqui)

Uma versão resumida deste documento está disponível em `README_en.md` para facilitar contribuições internacionais.

## Arquitetura

Cada arquivo em `devai/` possui uma responsabilidade específica:
- `__init__.py` – inicialização do pacote.
- `__main__.py` – ponto de entrada para `python -m devai`.
- `ai_model.py` – comunicação segura com o modelo de linguagem.
- `analyzer.py` – análise do código do projeto e grafo de dependências.
- `api_schemas.py` – modelos Pydantic usados pela API.
- `approval.py` – fila de solicitações de aprovação.
- `auto_review.py` – geração de revisões automáticas.
- `cli.py` – interface de linha de comando baseada em Rich.
- `command_router.py` – roteamento dos comandos da CLI.
- `complexity_tracker.py` – registro histórico da complexidade do projeto.
- `config.py` – carregamento de configuração e logger.
- `conversation_handler.py` – gerenciamento de múltiplas sessões de conversa.
- `core.py` – orquestrador principal e servidor FastAPI.
- `decision_log.py` – armazenamento de decisões aprovadas.
 - `dependency_check.py` – verificação de dependências obrigatórias.
- `dialog_summarizer.py` – sumarização de diálogos longos.
- `error_handler.py` – persistência de erros.
- `feedback.py` – registro de feedback para RLHF.
- `file_history.py` – controle de histórico de arquivos editados.
- `intent_classifier.py` – classificador de intenções do usuário.
- `intent_router.py` – despacho de comandos conforme a intenção.
- `learning_engine.py` – consolidação de lições aprendidas.
- `lint.py` – verificação rápida de TODOs e afins.
- `log_monitor.py` – monitoramento de logs de execução.
- `memory.py` – base vetorial de memórias e busca.
- `metacognition.py` – avaliação contínua de desempenho.
- `monitor_engine.py` – ciclo de monitoramento automático.
- `notifier.py` – envio opcional de notificações.
- `patch_utils.py` – utilidades para aplicar patches.
- `unidiff/` – parser minimalista para patches unificados.
- `plugin_manager.py` – carregamento de plugins externos.
- `prompt_engine.py` – construção dinâmica de prompts.
- `prompt_utils.py` – funções auxiliares para prompts.
- `rlhf.py` – rotina de fine‑tuning por feedback.
- `sandbox.py` – execução isolada de comandos.
- `shadow_mode.py` – simulação de alterações sem aplicar.
- `symbolic_memory_tagger.py` – marcação simbólica de memórias.
- `symbolic_training.py` – treinamento a partir de regras simbólicas.
- `symbolic_verification.py` – verificação de invariantes simbólicos.
- `tasks.py` – gerenciador de tarefas automatizadas.
- `test_runner.py` – executor de testes do projeto.
- `tui.py` – interface textual opcional.
- `ui.py` – pequena interface web.
- `update_manager.py` – aplicação de mudanças com rollback.
