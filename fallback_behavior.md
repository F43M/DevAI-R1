# Comportamento de Fallback para /analyze_deep

Caso o modelo não retorne a marcação `===RESPOSTA===` separando plano de resposta,
o backend exibirá o texto completo ao usuário e registrará um aviso no log
`ai_core.log` informando que a separação falhou. Isso garante que a informação
chegue ao usuário mesmo sem estrutura adequada.
