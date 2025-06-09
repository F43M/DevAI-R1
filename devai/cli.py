import asyncio
import json

from .config import config, logger
from .error_handler import friendly_message, log_error
from .core import CodeMemoryAI, run_scheduled_rlhf
from .feedback import FeedbackDB, registrar_preferencia
from .decision_log import log_decision
from pathlib import Path
import re


async def cli_main(guided: bool = False):
    """Interactive command loop for DevAI.

    Comandos principais: /lembrar, /esquecer, /ajustar, /rastrear e /memoria.
    """
    print("Inicializando CodeMemoryAI com DeepSeek-R1...")
    ai = CodeMemoryAI()
    feedback_db = FeedbackDB()
    run_scan = False
    if config.START_MODE == "full":
        run_scan = True
    elif config.START_MODE == "custom" and "scan" in config.START_TASKS:
        run_scan = True
    if run_scan:
        asyncio.create_task(ai.analyzer.deep_scan_app())
    else:
        print("Deep scan adiado para /deep_analysis")
    flag = Path.home() / ".devai_first_cli"
    if guided or not flag.exists():
        print("\n👋 Bem-vindo ao DevAI!")
        print("1. Abra um arquivo com /abrir <arquivo>")
        print("2. Faça uma pergunta ou solicite uma melhoria")
        print("3. Veja a resposta gerada")
        print("Dica: rode novamente com --guided para mais explicações")
        try:
            flag.touch()
        except Exception:
            pass
    print("\nDev IA Avançado Pronto!")
    print("Comandos disponíveis:")
    print("/memoria tipo:<tag> [filtro] --detalhado - Busca memórias")
    print("/lembrar <conteúdo> tipo:<tag> - Armazena memória")
    print("/esquecer <termo> - Desativa memórias")
    print("/ajustar estilo:<param> valor:<opcao> - Ajusta preferência")
    print("/rastrear <arquivo|tarefa> - Mostra histórico")
    print("/tarefa <nome> [args] - Executa uma tarefa")
    print("/analisar <função> - Analisa impacto de mudanças")
    print("/verificar - Verifica conformidade com especificação")
    print("/grafo - Mostra grafo de dependências")
    print("/ls [caminho] - Lista arquivos e pastas")
    print("/abrir <arquivo> [ini] [fim] - Mostra linhas do arquivo")
    print("/editar <arquivo> <linha> <novo> - Edita linha do arquivo")
    print("/novoarq <arquivo> [conteudo] - Cria novo arquivo")
    print("/novapasta <caminho> - Cria nova pasta")
    print("/deletar <caminho> - Remove arquivo ou pasta")
    print("/historico <arquivo> - Mostra histórico de mudanças")
    print("/feedback <arquivo> <tag> <motivo> - Registrar feedback negativo")
    print("/sair - Encerra")
    try:
        while True:
            try:
                user_input = input("\n>>> ").strip()
                if user_input.lower() == "/sair":
                    break
                elif user_input.lower().startswith("/memoria"):
                    args = user_input[len("/memoria"):].strip()
                    detailed = "--detalhado" in args
                    if detailed:
                        args = args.replace("--detalhado", "").strip()
                    page = 1
                    m_page = re.search(r"page:(\d+)", args)
                    if m_page:
                        page = int(m_page.group(1))
                        args = args.replace(m_page.group(0), "").strip()
                    m_type = re.search(r"tipo:([\w_]+)", args)
                    memory_type = m_type.group(1) if m_type else None
                    if memory_type:
                        args = args.replace(m_type.group(0), "").strip()
                    query = args.replace("filtro:", "").strip()
                    results = ai.memory.search(query or "recent", top_k=5 * page, memory_type=memory_type)
                    results = results[(page - 1) * 5 : page * 5]
                    print("\nMemórias relevantes:")
                    for m in results:
                        text = m['content']
                        if query:
                            text = text.replace(query, f"**{query}**")
                        line = f"- [{m['similarity_score']:.2f}] {text[:80]}"
                        if detailed:
                            line += f" (id: {m['id']}, tags: {', '.join(m['tags'])})"
                        print(line)
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
                elif user_input.startswith("/novoarq "):
                    parts = user_input[len("/novoarq "):].split(maxsplit=1)
                    file = parts[0]
                    content = parts[1] if len(parts) > 1 else ""
                    ok = await ai.analyzer.create_file(file, content)
                    print("✅ Arquivo criado" if ok else "Falha ao criar")
                elif user_input.startswith("/novapasta "):
                    path = user_input[len("/novapasta "):].strip()
                    ok = await ai.analyzer.create_directory(path)
                    print("✅ Pasta criada" if ok else "Falha ao criar pasta")
                elif user_input.startswith("/deletar "):
                    path = user_input[len("/deletar "):].strip()
                    confirm = input("Tem certeza que deseja remover? [s/N] ").lower()
                    if confirm != "s":
                        print("Operação cancelada")
                        continue
                    ok = await ai.analyzer.delete_file(path)
                    if not ok:
                        ok = await ai.analyzer.delete_directory(path)
                    print("✅ Removido" if ok else "Falha ao remover")
                elif user_input.startswith("/lembrar "):
                    args = user_input[len("/lembrar "):].strip()
                    m_type = re.search(r"tipo:([\w_]+)", args)
                    memory_type = m_type.group(1) if m_type else None
                    if memory_type:
                        args = args.replace(m_type.group(0), "").strip()
                    ai.memory.save({
                        "type": "manual",
                        "content": args,
                        "metadata": {"source": "cli"},
                        "tags": [memory_type] if memory_type else [],
                        "memory_type": memory_type,
                    })
                    log_decision("comando", "lembrar", args, "cli", "ok")
                    print("✅ Memória registrada")
                elif user_input.startswith("/esquecer "):
                    term = user_input[len("/esquecer "):].strip()
                    count = ai.memory.deactivate_memories(term)
                    log_decision("comando", "esquecer", term, "cli", str(count))
                    print(f"Memórias desativadas: {count}")
                elif user_input.startswith("/ajustar "):
                    args = user_input[len("/ajustar "):].strip()
                    m_param = re.search(r"estilo:([\w_]+)", args)
                    m_val = re.search(r"valor:([\w_]+)", args)
                    if not (m_param and m_val):
                        print("Uso: /ajustar estilo:<par> valor:<opcao>")
                    else:
                        param = m_param.group(1)
                        val = m_val.group(1)
                        prefs = {}
                        if Path("PREFERENCES_STORE.json").exists():
                            prefs = json.loads(Path("PREFERENCES_STORE.json").read_text())
                        prefs[param] = val
                        Path("PREFERENCES_STORE.json").write_text(json.dumps(prefs, indent=2))
                        log_decision("comando", "ajustar", f"{param}={val}", "cli", "ok")
                        print("Preferência atualizada")
                elif user_input.startswith("/preferencia "):
                    text = user_input[len("/preferencia "):].strip().strip('"')
                    registrar_preferencia(text)
                    print("Preferência registrada com sucesso")
                elif user_input.startswith("/rastrear "):
                    target = user_input[len("/rastrear "):].strip()
                    print("-- Rastreamento --")
                    hist = await ai.analyzer.get_history(target)
                    for h in hist:
                        print(json.dumps(h, indent=2))
                    for h in ai.tasks.get_history():
                        if target in h.get("task", "") or target in str(h.get("args", "")):
                            print(json.dumps(h, indent=2))
                    log_path = Path("decision_log.yaml")
                    if log_path.exists():
                        try:
                            import yaml  # type: ignore
                        except Exception:  # pragma: no cover - fallback when PyYAML is missing
                            from . import yaml_fallback as yaml
                        data = yaml.safe_load(log_path.read_text()) or []
                        for e in data:
                            if target in e.get("modulo", "") or target in e.get("motivo", ""):
                                print(json.dumps(e, indent=2))
                    if Path("SELF_REFLECTION.md").exists():
                        for line in Path("SELF_REFLECTION.md").read_text().splitlines():
                            if target in line:
                                print(line.strip())
                    log_decision("comando", "rastrear", target, "cli", "ok")
                elif user_input == "/plugins":
                    for p in ai.tasks.plugins.list_plugins():
                        status = "on" if p["active"] else "off"
                        print(f"- {p['name']} ({status})")
                elif user_input.startswith("/plugin "):
                    parts = user_input.split()
                    if len(parts) != 3 or parts[2] not in {"on", "off"}:
                        print("Uso: /plugin <nome> on|off")
                    else:
                        name = parts[1]
                        if parts[2] == "on":
                            ok = ai.tasks.plugins.enable_plugin(name)
                            print("✅ Plugin ativado" if ok else "Plugin não encontrado")
                        else:
                            ok = ai.tasks.plugins.disable_plugin(name)
                            print("✅ Plugin desativado" if ok else "Plugin não encontrado")
                elif user_input.startswith("/historico "):
                    file = user_input[len("/historico "):].strip()
                    hist = await ai.analyzer.get_history(file)
                    for h in hist:
                        print(json.dumps(h, indent=2))
                elif user_input.startswith("/treinar_rlhf"):
                    parts = user_input.split()
                    if len(parts) < 2:
                        print("Uso: /treinar_rlhf <modelo_base> [destino]")
                    else:
                        base = parts[1]
                        out = parts[2] if len(parts) > 2 else "./model_ft"
                        from .rlhf import train_from_memory
                        result = await train_from_memory(base, out)
                        print(json.dumps(result, indent=2))
                elif user_input.startswith("/treinar_rlhf_auto"):
                    result = await run_scheduled_rlhf(ai.memory)
                    print(json.dumps(result, indent=2))
                elif user_input.startswith("/train_intents"):
                    path = user_input[len("/train_intents"):].strip() or "intent_samples.json"
                    file_path = Path(path)
                    if not file_path.exists():
                        print("Arquivo de exemplos não encontrado")
                    else:
                        data = json.loads(file_path.read_text())
                        samples = []
                        for item in data:
                            if isinstance(item, dict):
                                samples.append((item.get("text", ""), item.get("intent", "")))
                            elif isinstance(item, list) and len(item) == 2:
                                samples.append((item[0], item[1]))
                        from .intent_classifier import train_intent_model
                        train_intent_model(samples)
                        print("Modelo de intenções treinado")
                elif user_input.startswith("/feedback "):
                    parts = user_input[len("/feedback "):].split(maxsplit=2)
                    if len(parts) < 3:
                        print("Uso: /feedback <arquivo> <tag> <motivo>")
                    else:
                        feedback_db.add(parts[0], parts[1], parts[2])
                        print("Feedback registrado")
                else:
                    response = await ai.generate_response(user_input, double_check=ai.double_check)
                    print("\nResposta:")
                    print(response)
            except Exception as e:
                log_error("CLI", e)
                print(friendly_message(e))
    except KeyboardInterrupt:
        logger.info("🧼 Encerrando DevAI...")
    finally:
        if hasattr(ai, "shutdown"):
            await ai.shutdown()
