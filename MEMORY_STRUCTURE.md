# Estrutura da Memória

A tabela `memory` armazena entradas persistentemente.

Campos principais:
- `type`: categoria livre do evento
- `memory_type`: classificação simbólica (`explicacao`, `bug corrigido`, `feedback negativo`, `refatoracao aprovada`, `regra do usuario`, `licao aprendida`)
- `content`, `metadata`, `embedding`

Consultas podem filtrar por `memory_type` usando o comando `/memoria tipo:<tag>`.
