import asyncio
import json

from .config import config, logger
from .core import CodeMemoryAI


async def cli_main():
    print("Inicializando CodeMemoryAI com DeepSeek-R1...")
    ai = CodeMemoryAI()
    asyncio.create_task(ai.analyzer.deep_scan_app())
    print("\nDev IA Avançado Pronto!")
    print("Comandos disponíveis:")
    print("/memoria <query> - Busca memórias relevantes")
    print("/tarefa <nome> [args] - Executa uma tarefa")
    print("/analisar <função> - Analisa impacto de mudanças")
    print("/verificar - Verifica conformidade com especificação")
    print("/grafo - Mostra grafo de dependências")
    print("/ls [caminho] - Lista arquivos e pastas")
    print("/abrir <arquivo> [ini] [fim] - Mostra linhas do arquivo")
    print("/editar <arquivo> <linha> <novo> - Edita linha do arquivo")
    print("/sair - Encerra")
    while True:
        try:
            user_input = input("\n>>> ").strip()
            if user_input.lower() == "/sair":
                break
            elif user_input.lower().startswith("/memoria"):
                query = user_input[len("/memoria"):].strip() or "recent"
                memories = ai.memory.search(query, top_k=5)
                print("\nMemórias relevantes:")
                for m in memories:
                    print(f"- [{m['similarity_score']:.2f}] {m['content'][:80]}... (tags: {', '.join(m['tags'])})")
            elif user_input.startswith("/tarefa "):
                parts = user_input[len("/tarefa "):].split()
                task_name = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                result = await ai.tasks.run_task(task_name, *args)
                print(json.dumps(result, indent=2))
            elif user_input.startswith("/analisar "):
                func = user_input[len("/analisar "):]
                report = await ai.analyze_impact([func])
                for item in report:
                    print(f"\nImpacto em {item['target']} (gatilhos: {', '.join(item['triggers'])}):")
                    for finding in item["findings"]:
                        if isinstance(finding, dict):
                            print(f"- {finding.get('chunk')}:")
                            for issue in finding.get('issues', []):
                                print(f"  {issue}")
                        else:
                            print(f"- {finding}")
            elif user_input == "/verificar":
                spec = {"calculate_score": {"expected_inputs": ["data", "weights"], "expected_output": "number"}}
                findings = await ai.verify_compliance(spec)
                for finding in findings:
                    print(finding)
            elif user_input == "/grafo":
                graph = ai.analyzer.get_code_graph()
                print("Grafo de dependências:")
                for node in graph["nodes"]:
                    print(f"- {node['id']} ({node.get('type', 'function')})")
                print("\nConexões:")
                for link in graph["links"]:
                    print(f"{link['source']} -> {link['target']}")
            elif user_input.startswith("/ls"):
                path = user_input[len("/ls"):].strip()
                items = await ai.analyzer.list_dir(path)
                for item in items:
                    print(item)
            elif user_input.startswith("/abrir "):
                parts = user_input[len("/abrir "):].split()
                file = parts[0]
                start = int(parts[1]) if len(parts) > 1 else 1
                end = int(parts[2]) if len(parts) > 2 else start
                lines = await ai.analyzer.read_lines(file, start, end)
                for i, line in enumerate(lines, start=start):
                    print(f"{i}: {line}")
            elif user_input.startswith("/editar "):
                parts = user_input[len("/editar "):].split(maxsplit=2)
                if len(parts) < 3:
                    print("Uso: /editar <arquivo> <linha> <novo>")
                else:
                    file, line_no, new_line = parts[0], int(parts[1]), parts[2]
                    ok = await ai.analyzer.edit_line(file, line_no, new_line)
                    print("✅ Linha atualizada" if ok else "Falha ao editar")
            else:
                response = await ai.generate_response(user_input)
                print("\nResposta:")
                print(response)
        except Exception as e:
            logger.error("Erro na CLI", input=user_input, error=str(e))
            print(f"Erro: {str(e)}")
