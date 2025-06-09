# DevAI-R1

Assistente de desenvolvimento baseado em IA com suporte a contextos de até **160k tokens** via OpenRouter. Agora inclui validação de configuração e autenticação com tokens JWT.

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

3. Instale as dependências do projeto para habilitar a comunicação real com o OpenRouter. A nova interface interativa usa `rich` e `prompt_toolkit`:

```bash
pip install -r requirements.txt
```

4. (Opcional) Descreva comandos de build e testes adicionais em `PROJECT_GUIDE.md`.

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

Você pode ajustar no `config.yaml`, via `--approval-mode` ao iniciar
ou dinamicamente com o comando `/modo`.

O DevAI traz versões simplificadas de algumas bibliotecas (como `aiohttp` e `fastapi`) usadas apenas em testes offline. O módulo `dependency_check` avisará caso essas versões estejam ativas, recomendando a instalação dos pacotes reais.

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
- `/modo <suggest|auto_edit|full_auto>` altera o nível de aprovação em tempo real.
- `/ajuda` exibe a documentação completa de comandos.

Ao usar `/deletar`, a CLI exibe um diálogo de confirmação para evitar remoções acidentais.

### Histórico de complexidade

Com o servidor API rodando é possível consultar `/complexity/history` para obter
o histórico de variações de complexidade salvas em `complexity_history.json`.
O endpoint retorna uma lista de registros e a tendência média recente:

```json
{
  "history": [{"timestamp": "2024-01-01T12:00:00", "average_complexity": 3.2}],
  "trend": -0.1
}
```

Valores negativos de `trend` indicam redução da complexidade média do projeto.

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

## Treinamento RLHF

Depois de registrar feedback positivo via API ou CLI, é possível refinar o modelo base utilizando a biblioteca [`trl`](https://github.com/huggingface/trl).
Instale as dependências opcionais `transformers` e `trl` e execute:

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
- Dependências opcionais com fallback *(implementado)*
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
