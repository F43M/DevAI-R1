# Learning Engine

O `learning_engine.py` coordena sessões de aprendizado simbólico com apoio do modelo DeepSeek R1. Ele percorre o código, analisa erros e extrai padrões positivos, registrando as descobertas no `MemoryManager`.

Principais funções:

- **learn_from_codebase()** – Para cada função conhecida gera explicações, riscos e boas práticas.
- **learn_from_errors()** – Analisa logs de erro e registra lições aprendidas.
- **extract_positive_patterns()** – Usa trechos marcados como refatoração aprovada para gerar padrões reutilizáveis.
- **reflect_on_internal_knowledge()** – Resume o conhecimento recente e salva em `logs/learning_report.md`.
- **import_external_codebase(path)** – Executa o mesmo processo em outro repositório, salvando as memórias como `aprendizado_importado`.

As chamadas ao modelo respeitam um limite simples de requisições por minuto e todas as respostas são armazenadas como memórias classificadas.
