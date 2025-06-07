"""Script de teste simples para o painel DevAI."""

import asyncio
from pathlib import Path
from devai.core import CodeMemoryAI


async def main() -> None:
    ai = CodeMemoryAI()

    # upload de codigo
    tmp = Path(ai.analyzer.code_root) / "painel_tmp.py"
    tmp.write_text("def foo():\n    return 1\n")
    await ai.analyzer.parse_file(tmp)
    print("Arquivo carregado")

    # analise com pensamento longo
    resp = await ai.generate_response("Analise a funcao foo", double_check=True)
    print("Resposta:", resp.splitlines()[0])

    # execucao de tarefa
    await ai.tasks.run_task("run_tests")
    print("Tarefa de testes executada")

    # refatoracao com rollback
    result = await ai.tasks.run_task("auto_refactor", str(tmp))
    print("Resultado da refatoracao:", result)

    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
