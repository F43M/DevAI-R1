# Limitações conhecidas

A configuração de `max_tokens` e `temperature` pode ser ignorada por alguns
provedores de modelo. Caso isso ocorra, o comportamento do endpoint
`/analyze_deep` poderá variar em tamanho de resposta. Ainda assim o código
utiliza 4096 tokens e temperatura 0.2 como valores desejados.
