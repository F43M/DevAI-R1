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
