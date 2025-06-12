# Estrutura da Memória

A tabela `memory` armazena entradas persistentemente.

Campos principais:
- `type`: categoria livre do evento
- `memory_type`: classificação simbólica (`explicacao`, `bug corrigido`, `feedback negativo`, `refatoracao aprovada`, `regra do usuario`, `licao aprendida`)
- `content`, `metadata`, `embedding`

Consultas podem filtrar por `memory_type` usando o comando `/memoria tipo:<tag>`.

Tags associadas a cada entrada ficam na tabela `tags`. Sempre que uma tag é
registrada, seu contador em `tag_stats` é incrementado, permitindo analisar
frequências de uso.

Exemplos de tags geradas automaticamente:
- `@nova_funcao`, `@refatorado` – controle de versões
- `@complexo` – funções com complexidade acima do limiar definido em
  `COMPLEXITY_TAG_THRESHOLD`
- `@descontinuado` – detectado em docstrings contendo “deprecated” ou
  “obsoleto”

O índice vetorial usado na busca é salvo em `faiss.index` junto com o arquivo
`faiss_ids.json` contendo os IDs das entradas. Ambos são carregados ao iniciar o
`MemoryManager`, evitando reindexar todo o banco. Caso deseje, você pode mudar
esses caminhos no `config.yaml`.
