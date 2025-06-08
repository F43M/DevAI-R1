# Configuração do DevAI

Todos os parâmetros de modelo devem ser definidos apenas em `MODELS.default` dentro do `config.yaml`:

```yaml
MODELS:
  default:
    name: deepseek/deepseek-r1-0528:free
    api_key: ${OPENROUTER_API_KEY}
    url: https://openrouter.ai/api/v1/chat/completions
```

O campo `MODEL_NAME` está **obsoleto** e será ignorado futuramente. Utilize `config.model_name` para obter o nome ativo do modelo em código.

