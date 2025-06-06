# DevAI-R1
Sistema de inteligência de desenvolvimento com IAs.

Esta atualização permite explorar todo o limite de 160k tokens
oferecido pela API do OpenRouter para entrada e saída de dados.
Agora o aplicativo define `MAX_CONTEXT_LENGTH` em `160000` e envia
prompts para a API utilizando esse limite, garantindo respostas mais
longas quando necessário.

## Novidades

- **Persistência de índice FAISS**: o índice de busca semântica é salvo em disco,
  acelerando a inicialização.
- **Limpeza automática de memórias**: entradas antigas ou pouco acessadas são
  removidas diariamente para manter o banco enxuto.
- **Tempo limite nas chamadas à API**: requisições ao OpenRouter agora possuem
  timeout e contagem de erros para maior estabilidade.
- **Métricas de uso**: número de chamadas à API e tempo médio de resposta estão
  disponíveis em `/metrics`.

## Configuração

Crie um arquivo `config.yaml` na raiz para personalizar caminhos e outras
configurações. Exemplo:

```yaml
CODE_ROOT: ./app
API_PORT: 8000
```

Variáveis de ambiente ainda podem sobrescrever `OPENROUTER_API_KEY`.

## Monitoramento da pasta `./app`

O aplicativo verifica periodicamente a pasta especificada em `CODE_ROOT` e se
atualiza automaticamente sempre que novos arquivos Python são adicionados ou
modificados.

Uma interface de administração em `/admin` permite acionar uma nova varredura
manual e exibe informações sobre o último escaneamento.
