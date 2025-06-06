# DevAI-R1

Development assistant powered by AI with support for up to **160k tokens** via OpenRouter.

## Main features

- **Persistent memory** with FAISS vector search
- **Context levels** for short, medium and long term memory
- **Embedding cache** to accelerate repeated queries
- **Automatic memory cleanup** and usage feedback
- **Automated testing and static analysis tasks**
- **Coverage reports**

The project exposes a CLI and a web API. See `README.md` for detailed commands in Portuguese.

## Tests

Install the development dependencies and run:

```bash
pytest
```

You can also execute tests and static analysis through the task manager:

```bash
python -m devai --cli
/tarefa run_tests
/tarefa static_analysis
```
