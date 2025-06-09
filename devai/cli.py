import asyncio
import json

from .config import config, logger
from .error_handler import friendly_message, log_error
from .core import CodeMemoryAI, run_scheduled_rlhf
from .feedback import FeedbackDB, registrar_preferencia
from .decision_log import log_decision
from pathlib import Path
import re
import tempfile
from typing import Dict

from .update_manager import UpdateManager

from .ui import CLIUI
try:
    from .tui import TUIApp
except Exception:  # pragma: no cover - optional dependency
    TUIApp = None  # type: ignore
from rich.panel import Panel


async def handle_sair(ai, ui, args, *, plain, feedback_db):
    """Handle /sair command."""
    return True


async def handle_ajuda(ai, ui, args, *, plain, feedback_db):
    help_path = Path(__file__).resolve().parent.parent / "COMMANDS_REFERENCE.md"
    if help_path.exists():
        print(help_path.read_text())
    else:
        print("Arquivo de ajuda não encontrado")


async def handle_memoria(ai, ui, args, *, plain, feedback_db):
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
        text = m["content"]
        if query:
            text = text.replace(query, f"**{query}**")
        line = f"- [{m['similarity_score']:.2f}] {text[:80]}"
        if detailed:
            line += f" (id: {m['id']}, tags: {', '.join(m['tags'])})"
        print(line)


async def handle_tarefa(ai, ui, args, *, plain, feedback_db):
    parts = args.split()
    if not parts:
        return
    task_name = parts[0]
    task_args = parts[1:] if len(parts) > 1 else []
    if task_name in {"run_tests", "coverage"}:
        async with ui.progress("executando tarefa...") as update:
            result = await ai.tasks.run_task(task_name, *task_args, progress=update)
    else:
        result = await ai.tasks.run_task(task_name, *task_args)
    print(json.dumps(result, indent=2))


async def handle_analisar(ai, ui, args, *, plain, feedback_db):
    report = await ai.analyze_impact([args])
    for item in report:
        print(f"\nImpacto em {item['target']} (gatilhos: {', '.join(item['triggers'])}):")
        for finding in item["findings"]:
            if isinstance(finding, dict):
                print(f"- {finding.get('chunk')}:")
                for issue in finding.get("issues", []):
                    print(f"  {issue}")
            else:
                print(f"- {finding}")


async def handle_verificar(ai, ui, args, *, plain, feedback_db):
    spec = {
        "calculate_score": {
            "expected_inputs": ["data", "weights"],
            "expected_output": "number",
        }
    }
    findings = await ai.verify_compliance(spec)
    for finding in findings:
        print(finding)


async def handle_grafo(ai, ui, args, *, plain, feedback_db):
    graph = ai.analyzer.get_code_graph()
    print("Grafo de dependências:")
    for node in graph["nodes"]:
        print(f"- {node['id']} ({node.get('type', 'function')})")
    print("\nConexões:")
    for link in graph["links"]:
        print(f"{link['source']} -> {link['target']}")


async def handle_ls(ai, ui, args, *, plain, feedback_db):
    items = await ai.analyzer.list_dir(args.strip())
    for item in items:
        print(item)


async def handle_abrir(ai, ui, args, *, plain, feedback_db):
    parts = args.split()
    if not parts:
        return
    file = parts[0]
    start = int(parts[1]) if len(parts) > 1 else 1
    end = int(parts[2]) if len(parts) > 2 else start
    lines = await ai.analyzer.read_lines(file, start, end)
    for i, line in enumerate(lines, start=start):
        print(f"{i}: {line}")


async def handle_editar(ai, ui, args, *, plain, feedback_db):
    parts = args.split(maxsplit=2)
    if len(parts) < 3:
        print("Uso: /editar <arquivo> <linha> <novo>")
        return
    file, line_no, new_line = parts[0], int(parts[1]), parts[2]
    ok = await ai.analyzer.edit_line(file, line_no, new_line)
    print("✅ Linha atualizada" if ok else "Falha ao editar")


async def handle_novoarq(ai, ui, args, *, plain, feedback_db):
    parts = args.split(maxsplit=1)
    if not parts:
        return
    file = parts[0]
    content = parts[1] if len(parts) > 1 else ""
    ok = await ai.analyzer.create_file(file, content)
    print("✅ Arquivo criado" if ok else "Falha ao criar")


async def handle_novapasta(ai, ui, args, *, plain, feedback_db):
    path = args.strip()
    ok = await ai.analyzer.create_directory(path)
    print("✅ Pasta criada" if ok else "Falha ao criar pasta")


async def handle_deletar(ai, ui, args, *, plain, feedback_db):
    path = args.strip()
    confirmed = await ui.confirm("Tem certeza que deseja remover?")
    if not confirmed:
        print("Operação cancelada")
        return
    ok = await ai.analyzer.delete_file(path)
    if not ok:
        ok = await ai.analyzer.delete_directory(path)
    print("✅ Removido" if ok else "Falha ao remover")


async def handle_lembrar(ai, ui, args, *, plain, feedback_db):
    m_type = re.search(r"tipo:([\w_]+)", args)
    memory_type = m_type.group(1) if m_type else None
    if memory_type:
        args = args.replace(m_type.group(0), "").strip()
    ai.memory.save(
        {
            "type": "manual",
            "content": args,
            "metadata": {"source": "cli"},
            "tags": [memory_type] if memory_type else [],
            "memory_type": memory_type,
        }
    )
    log_decision("comando", "lembrar", args, "cli", "ok")
    print("✅ Memória registrada")


async def handle_esquecer(ai, ui, args, *, plain, feedback_db):
    term = args.strip()
    count = ai.memory.deactivate_memories(term)
    log_decision("comando", "esquecer", term, "cli", str(count))
    print(f"Memórias desativadas: {count}")


async def handle_ajustar(ai, ui, args, *, plain, feedback_db):
    m_param = re.search(r"estilo:([\w_]+)", args)
    m_val = re.search(r"valor:([\w_]+)", args)
    if not (m_param and m_val):
        print("Uso: /ajustar estilo:<par> valor:<opcao>")
        return
    param = m_param.group(1)
    val = m_val.group(1)
    prefs = {}
    if Path("PREFERENCES_STORE.json").exists():
        prefs = json.loads(Path("PREFERENCES_STORE.json").read_text())
    prefs[param] = val
    Path("PREFERENCES_STORE.json").write_text(json.dumps(prefs, indent=2))
    log_decision("comando", "ajustar", f"{param}={val}", "cli", "ok")
    print("Preferência atualizada")


async def handle_preferencia(ai, ui, args, *, plain, feedback_db):
    text = args.strip().strip('"')
    registrar_preferencia(text)
    print("Preferência registrada com sucesso")


async def handle_rastrear(ai, ui, args, *, plain, feedback_db):
    target = args.strip()
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


async def handle_plugins(ai, ui, args, *, plain, feedback_db):
    for p in ai.tasks.plugins.list_plugins():
        status = "on" if p["active"] else "off"
        print(f"- {p['name']} ({status})")


async def handle_plugin(ai, ui, args, *, plain, feedback_db):
    parts = ["/plugin"] + args.split()
    if len(parts) != 3 or parts[2] not in {"on", "off"}:
        print("Uso: /plugin <nome> on|off")
        return
    name = parts[1]
    if parts[2] == "on":
        ok = ai.tasks.plugins.enable_plugin(name)
        print("✅ Plugin ativado" if ok else "Plugin não encontrado")
    else:
        ok = ai.tasks.plugins.disable_plugin(name)
        print("✅ Plugin desativado" if ok else "Plugin não encontrado")


async def handle_historia(ai, ui, args, *, plain, feedback_db):
    session_id = args.strip() or "default"
    hist = ai.conv_handler.history(session_id)
    for m in hist:
        if plain:
            print(f"{m.get('role')}: {m.get('content')}")
        else:
            ui.console.print(Panel(m.get("content", ""), title=m.get("role", ""), expand=False))


async def handle_historico(ai, ui, args, *, plain, feedback_db):
    file = args.strip()
    hist = await ai.analyzer.get_history(file)
    for h in hist:
        print(json.dumps(h, indent=2))


async def handle_historico_cli(ai, ui, args, *, plain, feedback_db):
    arg = args.strip()
    num = int(arg) if arg.isdigit() else None
    hist = ui.get_log()
    if num is not None:
        hist = hist[-num:]
    for line in hist:
        print(line)


async def handle_treinar_rlhf(ai, ui, args, *, plain, feedback_db):
    parts = ["/treinar_rlhf"] + args.split()
    if len(parts) < 2:
        print("Uso: /treinar_rlhf <modelo_base> [destino]")
        return
    base = parts[1]
    out = parts[2] if len(parts) > 2 else "./model_ft"
    from .rlhf import train_from_memory

    result = await train_from_memory(base, out)
    print(json.dumps(result, indent=2))


async def handle_treinar_rlhf_auto(ai, ui, args, *, plain, feedback_db):
    result = await run_scheduled_rlhf(ai.memory)
    print(json.dumps(result, indent=2))


async def handle_train_intents(ai, ui, args, *, plain, feedback_db):
    path = args.strip() or "intent_samples.json"
    file_path = Path(path)
    if not file_path.exists():
        print("Arquivo de exemplos não encontrado")
        return
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


async def handle_feedback(ai, ui, args, *, plain, feedback_db):
    parts = args.split(maxsplit=2)
    if len(parts) < 3:
        print("Uso: /feedback <arquivo> <tag> <motivo>")
    else:
        feedback_db.add(parts[0], parts[1], parts[2])
        print("Feedback registrado")


async def handle_refatorar(ai, ui, args, *, plain, feedback_db):
    target = args.strip()
    result = await ai.tasks.run_task("auto_refactor", target)
    print(json.dumps(result, indent=2))


async def handle_rever(ai, ui, args, *, plain, feedback_db):
    target = args.strip()
    result = await ai.tasks.run_task("code_review", target)
    for item in result:
        if isinstance(item, str):
            print(item)
        else:
            print(json.dumps(item, indent=2))


async def handle_resetar(ai, ui, args, *, plain, feedback_db):
    ai.conv_handler.reset("default")
    print("Conversa resetada.")


async def handle_tests_local(ai, ui, args, *, plain, feedback_db):
    cfg_path = Path("config.yaml")
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - fallback when PyYAML is missing
        from . import yaml_fallback as yaml
    data = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}
    new_val = not data.get("TESTS_USE_ISOLATION", config.TESTS_USE_ISOLATION)
    data["TESTS_USE_ISOLATION"] = new_val
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True))
    config.TESTS_USE_ISOLATION = new_val
    status = "ativada" if new_val else "desativada"
    print(f"Execução isolada {status}")


def _split_diff_by_file(diff: str) -> Dict[str, str]:
    """Return a mapping of file path to its diff chunk."""
    files: Dict[str, list[str]] = {}
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            if current and current in files:
                files[current].append("\n")
            current = None
        if line.startswith("+++ "):
            current = line.split()[1]
            if current.startswith("b/"):
                current = current[2:]
            files[current] = []
            continue
        if line.startswith("--- "):
            continue
        if current:
            files[current].append(line)
    return {k: "\n".join(v) for k, v in files.items()}


def _apply_patch_to_file(path: Path, diff: str) -> None:
    """Apply a unified diff chunk to a file."""
    lines = path.read_text().splitlines(keepends=True)
    result: list[str] = []
    i = 0
    diff_lines = diff.splitlines()
    idx = 0
    while idx < len(diff_lines):
        line = diff_lines[idx]
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if not m:
                idx += 1
                continue
            start = int(m.group(1)) - 1
            result.extend(lines[i:start])
            i = start
            idx += 1
            while idx < len(diff_lines):
                h = diff_lines[idx]
                if h.startswith("@@"):
                    break
                if h.startswith("-"):
                    i += 1
                elif h.startswith("+"):
                    result.append(h[1:] + "\n")
                else:
                    if i < len(lines):
                        result.append(lines[i])
                    i += 1
                idx += 1
            continue
        idx += 1
    result.extend(lines[i:])
    path.write_text("".join(result))


async def handle_default(ai, ui, args, *, plain, feedback_db):
    print("\nResposta:")
    tokens: list[str] = []
    async with ui.loading("Gerando resposta..."):
        async for token in ai.generate_response_stream(args):
            tokens.append(token)
            if plain:
                print(token, end="", flush=True)
            else:
                ui.console.print(token, end="")
    response = "".join(tokens)
    print()
    is_patch = bool(
        re.search(r"\ndiff --git", response)
        or re.search(r"^[+-](?![+-])", response, re.MULTILINE)
    )
    if is_patch:
        ui.render_diff(response)
        apply = await ui.confirm("Aplicar mudanças?")
        if apply:
            patches = _split_diff_by_file(response)
            updater = UpdateManager()
            for f, diff_text in patches.items():
                def _apply(p: Path, d=diff_text) -> None:
                    _apply_patch_to_file(p, d)

                success = updater.safe_apply(f, _apply)
                if success:
                    ui.console.print(f"[green]✅ {f} atualizado[/green]")
                    log_decision("patch", f, "apply", "cli", "ok")
                else:
                    ui.console.print(f"[red]❌ Falha em {f}[/red]")
                    log_decision("patch", f, "apply", "cli", "falha")
        else:
            log_decision("patch", "all", "rejeitado", "cli", "nao")
    elif plain:
        print(response)
    else:
        ui.console.print(response)
    ui.add_history(response)
    ui.show_history()


COMMANDS = {
    "sair": handle_sair,
    "ajuda": handle_ajuda,
    "memoria": handle_memoria,
    "tarefa": handle_tarefa,
    "analisar": handle_analisar,
    "verificar": handle_verificar,
    "grafo": handle_grafo,
    "ls": handle_ls,
    "abrir": handle_abrir,
    "editar": handle_editar,
    "novoarq": handle_novoarq,
    "novapasta": handle_novapasta,
    "deletar": handle_deletar,
    "lembrar": handle_lembrar,
    "esquecer": handle_esquecer,
    "ajustar": handle_ajustar,
    "preferencia": handle_preferencia,
    "rastrear": handle_rastrear,
    "plugins": handle_plugins,
    "plugin": handle_plugin,
    "historia": handle_historia,
    "historico": handle_historico,
    "historico_cli": handle_historico_cli,
    "treinar_rlhf": handle_treinar_rlhf,
    "treinar_rlhf_auto": handle_treinar_rlhf_auto,
    "train_intents": handle_train_intents,
    "feedback": handle_feedback,
    "refatorar": handle_refatorar,
    "rever": handle_rever,
    "resetar": handle_resetar,
    "tests_local": handle_tests_local,
}


async def cli_main(
    guided: bool = False,
    plain: bool = False,
    log: bool = True,
    tui: bool = False,
):
    """Interactive command loop for DevAI.

    Comandos principais: /lembrar, /esquecer, /ajustar, /rastrear e /memoria.
    """
    print("Inicializando CodeMemoryAI com DeepSeek-R1...")
    ai = CodeMemoryAI()
    feedback_db = FeedbackDB()
    cmd_list = [f"/{c}" for c in COMMANDS]
    ui = CLIUI(plain=plain, commands=cmd_list, log=log)
    ui.load_history()
    if tui and not plain and TUIApp is not None:
        app = TUIApp(ai=ai, cli_ui=ui, log=log)
        app.run()
        return
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
    print("/historia [sessao] - Exibe histórico de conversa")
    print("/historico <arquivo> - Mostra histórico de mudanças")
    print("/historico_cli [N] - Exibe N linhas do log da CLI (ou tudo)")
    print("/ajuda - Mostra documentação dos comandos")
    print("/feedback <arquivo> <tag> <motivo> - Registrar feedback negativo")
    print("/refatorar <arquivo> - Refatora o arquivo informado")
    print("/rever <arquivo> - Executa revisão de código")
    print("/resetar - Limpa o histórico de conversa")
    print("/tests_local - Alterna execução isolada dos testes")
    print("/sair - Encerra")
    try:
        while True:
            try:
                user_input = await ui.read_command("\n>>> ")
                ui.add_history(f">>> {user_input}")
                if user_input.startswith("/"):
                    parts = user_input[1:].split(" ", 1)
                    cmd = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    handler = COMMANDS.get(cmd)
                    if handler:
                        should_exit = await handler(ai, ui, args, plain=plain, feedback_db=feedback_db)
                        if should_exit:
                            break
                        continue
                await handle_default(ai, ui, user_input, plain=plain, feedback_db=feedback_db)
            except Exception as e:
                log_error("CLI", e)
                print(friendly_message(e))
    except KeyboardInterrupt:
        logger.info("🧼 Encerrando DevAI...")
    finally:
        if hasattr(ai, "shutdown"):
            await ai.shutdown()
