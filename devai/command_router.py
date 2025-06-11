# Command routing and handlers shared between CLI and TUI
import asyncio
import json
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from .patch_utils import apply_patch_to_file, split_diff_by_file

from rich.panel import Panel

from .config import config, logger
from .core import CodeMemoryAI, run_scheduled_rlhf
from .decision_log import log_decision, suggest_rules
from .feedback import FeedbackDB, registrar_preferencia

try:
    from .cli import UpdateManager  # type: ignore
except Exception:  # pragma: no cover - fallback for tests
    from .update_manager import UpdateManager
from . import approval
from .approval import requires_approval, request_approval


def _new_updater():
    from .cli import UpdateManager as UM

    return UM()


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
    results = ai.memory.search(
        query or "recent", top_k=5 * page, memory_type=memory_type
    )
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
            result = await ai.tasks.run_task(
                task_name, *task_args, progress=update, ui=ui
            )
    else:
        result = await ai.tasks.run_task(task_name, *task_args, ui=ui)
    print(json.dumps(result, indent=2))


async def handle_analisar(ai, ui, args, *, plain, feedback_db):
    report = await ai.analyze_impact([args])
    for item in report:
        print(
            f"\nImpacto em {item['target']} (gatilhos: {', '.join(item['triggers'])}):"
        )
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
    confirm = True
    if requires_approval("edit", file):
        if ui:
            confirm = await ui.confirm("Aplicar edição no arquivo?")
            model = "cli"
        else:
            confirm = await request_approval("Aplicar edição no arquivo?")
            model = "web"
        log_decision(
            "edit",
            file,
            "editar",
            model,
            "ok" if confirm else "nao",
            remember=getattr(ui, "remember_choice", False),
            expires_at=getattr(ui, "remember_expires", None),
        )
    if not confirm:
        print("Operação cancelada")
        return
    ok = await ai.analyzer.edit_line(file, line_no, new_line)
    print("✅ Linha atualizada" if ok else "Falha ao editar")


async def handle_novoarq(ai, ui, args, *, plain, feedback_db):
    parts = args.split(maxsplit=1)
    if not parts:
        return
    file = parts[0]
    content = parts[1] if len(parts) > 1 else ""
    confirm = True
    if requires_approval("create", file):
        if ui:
            confirm = await ui.confirm("Criar novo arquivo?")
            model = "cli"
        else:
            confirm = await request_approval("Criar novo arquivo?")
            model = "web"
        log_decision(
            "create",
            file,
            "novoarq",
            model,
            "ok" if confirm else "nao",
            remember=getattr(ui, "remember_choice", False),
            expires_at=getattr(ui, "remember_expires", None),
        )
    if not confirm:
        print("Operação cancelada")
        return
    ok = await ai.analyzer.create_file(file, content)
    print("✅ Arquivo criado" if ok else "Falha ao criar")


async def handle_novapasta(ai, ui, args, *, plain, feedback_db):
    path = args.strip()
    confirm = True
    if requires_approval("create", path):
        if ui:
            confirm = await ui.confirm("Criar nova pasta?")
            model = "cli"
        else:
            confirm = await request_approval("Criar nova pasta?")
            model = "web"
        log_decision(
            "create",
            path,
            "novapasta",
            model,
            "ok" if confirm else "nao",
            remember=getattr(ui, "remember_choice", False),
            expires_at=getattr(ui, "remember_expires", None),
        )
    if not confirm:
        print("Operação cancelada")
        return
    ok = await ai.analyzer.create_directory(path)
    print("✅ Pasta criada" if ok else "Falha ao criar pasta")


async def handle_deletar(ai, ui, args, *, plain, feedback_db):
    path = args.strip()
    confirmed = True
    if requires_approval("delete", path):
        if ui:
            confirmed = await ui.confirm("Tem certeza que deseja remover?")
            model = "cli"
        else:
            confirmed = await request_approval("Tem certeza que deseja remover?")
            model = "web"
        log_decision(
            "delete",
            path,
            "deletar",
            model,
            "ok" if confirmed else "nao",
            remember=getattr(ui, "remember_choice", False),
            expires_at=getattr(ui, "remember_expires", None),
        )
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


async def handle_decisoes(ai, ui, args, *, plain, feedback_db):
    """Show and manage decision log entries."""
    parts = args.split()
    log_path = Path("decision_log.yaml")
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - fallback when PyYAML is missing
        from . import yaml_fallback as yaml

    data = []
    if log_path.exists():
        try:
            data = yaml.safe_load(log_path.read_text()) or []
        except Exception:
            data = []
        if not isinstance(data, list):
            data = []

    # Toggle remembered approvals or purge expired
    if parts and parts[0] == "purge":
        now = datetime.now()
        kept = []
        removed = 0
        for e in data:
            exp = e.get("expires_at")
            if exp:
                try:
                    if datetime.fromisoformat(exp) < now:
                        removed += 1
                        continue
                except Exception:
                    pass
            kept.append(e)
        log_path.write_text(yaml.safe_dump(kept, allow_unicode=True))
        print(f"Expirações removidas: {removed}")
        return

    if len(parts) == 2 and parts[0] in {"lembrar", "esquecer"}:
        target_id = parts[1]
        for e in data:
            if str(e.get("id")) == target_id:
                e["remember"] = parts[0] == "lembrar"
                log_path.write_text(yaml.safe_dump(data, allow_unicode=True))
                print("Atualizado")
                return
        print("ID não encontrado")
        return

    # Parse optional filters
    count = 10
    if parts and parts[0].isdigit():
        count = int(parts[0])
        parts = parts[1:]
    rest = " ".join(parts)
    m_action = re.search(r"acao:([^\s]+)", rest)
    m_file = re.search(r"(?:arquivo|file):([^\s]+)", rest)
    action_filter = m_action.group(1) if m_action else None
    file_filter = m_file.group(1) if m_file else None

    entries = []
    for e in reversed(data):
        if action_filter and e.get("tipo") != action_filter:
            continue
        if file_filter and file_filter not in e.get("modulo", ""):
            continue
        entries.append(e)
        if len(entries) >= count:
            break

    for e in entries:
        flag = "*" if e.get("remember") else " "
        exp = e.get("expires_at")
        if exp:
            line = (
                f"{flag}[{e.get('id')}] {e.get('tipo')} {e.get('modulo')} - {e.get('motivo')} (até {exp})"
            )
        else:
            line = (
                f"{flag}[{e.get('id')}] {e.get('tipo')} {e.get('modulo')} - {e.get('motivo')}"
            )
        print(line)


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
            ui.console.print(
                Panel(m.get("content", ""), title=m.get("role", ""), expand=False)
            )


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
    result = await ai.tasks.run_task("auto_refactor", target, ui=ui)
    print(json.dumps(result, indent=2))


async def handle_rever(ai, ui, args, *, plain, feedback_db):
    target = args.strip()
    result = await ai.tasks.run_task("code_review", target, ui=ui)
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
    config.reload(str(cfg_path))
    status = "ativada" if new_val else "desativada"
    print(f"Execução isolada {status}")


async def handle_aprovar_proxima(ai, ui, args, *, plain, feedback_db):
    """Ativar aprovações automáticas temporárias."""
    try:
        count = int(args.strip() or "1")
    except ValueError:
        print("Uso: /aprovar_proxima <n>")
        return
    approval.auto_approve_remaining = max(0, count)
    print(
        f"Próximas {approval.auto_approve_remaining} ações aprovadas automaticamente"
    )


async def handle_aprovar_durante(ai, ui, args, *, plain, feedback_db):
    """Ativar aprovações automáticas por um período."""
    try:
        seconds = float(args.strip())
    except ValueError:
        print("Uso: /aprovar_durante <segundos>")
        return
    approval.auto_approve_until = datetime.now() + timedelta(seconds=seconds)
    print(
        f"Ações aprovadas automaticamente pelos próximos {int(seconds)} segundos"
    )


async def handle_modo(ai, ui, args, *, plain, feedback_db):
    """Alterar config.APPROVAL_MODE em tempo real."""
    mode = args.strip().lower()
    if mode not in {"suggest", "auto_edit", "full_auto"}:
        print("Uso: /modo <suggest|auto_edit|full_auto>")
        return
    old = config.APPROVAL_MODE
    config.APPROVAL_MODE = mode
    logger.info("Modo de aprovação alterado", antigo=old, novo=mode)
    log_decision("config", "APPROVAL_MODE", f"{old}->{mode}", "cli", "ok")


async def handle_regras(ai, ui, args, *, plain, feedback_db):
    """Gerenciar AUTO_APPROVAL_RULES via CLI."""
    parts = args.split()
    cfg_path = Path("config.yaml")
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - fallback when PyYAML is missing
        from . import yaml_fallback as yaml
    data = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}
    rules = data.get("AUTO_APPROVAL_RULES", [])

    if not parts:
        for i, r in enumerate(rules, start=1):
            status = "sim" if r.get("approve") else "nao"
            print(f"[{i}] {r.get('action')} {r.get('path')} -> {status}")
        if not rules:
            print("Nenhuma regra definida")
        return

    if parts[0] == "add" and len(parts) == 4:
        approve = parts[3].lower()
        if approve not in {"sim", "nao"}:
            print("Uso: /regras add <acao> <caminho> <sim|nao>")
            return
        rule = {
            "action": parts[1],
            "path": parts[2],
            "approve": approve == "sim",
        }
        rules.append(rule)
        data["AUTO_APPROVAL_RULES"] = rules
        cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True))
        config.reload(str(cfg_path))
        print("✅ Regra adicionada")
        return

    if parts[0] == "del" and len(parts) == 2:
        try:
            idx = int(parts[1]) - 1
            rules.pop(idx)
        except Exception:
            print("ID inválido")
            return
        data["AUTO_APPROVAL_RULES"] = rules
        cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True))
        config.reload(str(cfg_path))
        print("✅ Regra removida")
        return

    print("Uso: /regras [add <acao> <caminho> <sim|nao>|del <id>]")


async def handle_sugerir_regras(ai, ui, args, *, plain, feedback_db):
    """Mostrar sugestões de AUTO_APPROVAL_RULES baseadas no log."""
    parts = args.split()
    threshold = 3
    save = False
    for p in parts:
        if p.isdigit():
            threshold = int(p)
        if p == "--salvar":
            save = True
    suggestions = suggest_rules(threshold)
    if not suggestions:
        print("Nenhuma sugestão")
        return
    for i, r in enumerate(suggestions, start=1):
        print(f"[{i}] {r['action']} {r['path']} -> sim")
    if save:
        cfg_path = Path("config.yaml")
        try:
            import yaml  # type: ignore
        except Exception:  # pragma: no cover - fallback when PyYAML is missing
            from . import yaml_fallback as yaml
        data = {}
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text()) or {}
        rules = data.get("AUTO_APPROVAL_RULES", [])
        for r in suggestions:
            if r not in rules:
                rules.append(r)
        data["AUTO_APPROVAL_RULES"] = rules
        cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True))
        config.reload(str(cfg_path))
        print("✅ Regras atualizadas")




async def handle_default(
    ai, ui, args, *, plain, feedback_db, side_by_side: bool | None = None
):
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
        if side_by_side is None:
            side_by_side = config.DIFF_STYLE == "side_by_side"
        ui.render_diff(response, side_by_side=side_by_side)
        apply = True
        if requires_approval("patch"):
            if ui:
                apply = await ui.confirm("Aplicar mudanças?")
                model = "cli"
            else:
                apply = await request_approval(
                    "Aplicar mudanças?",
                    details=response,
                )
                model = "web"
        else:
            model = "cli"
        if apply:
            patches = split_diff_by_file(response)
            updater = _new_updater()
            for f, diff_text in patches.items():

                def _apply(p: Path, d=diff_text) -> None:
                    apply_patch_to_file(p, d)

                success = updater.safe_apply(f, _apply)
                if success:
                    ui.console.print(f"[green]✅ {f} atualizado[/green]")
                    log_decision(
                        "patch",
                        f,
                        "apply",
                        model,
                        "ok",
                        remember=ui.remember_choice,
                        expires_at=getattr(ui, "remember_expires", None),
                    )
                else:
                    ui.console.print(f"[red]❌ Falha em {f}[/red]")
                    log_decision(
                        "patch",
                        f,
                        "apply",
                        model,
                        "falha",
                        remember=ui.remember_choice,
                        expires_at=getattr(ui, "remember_expires", None),
                    )
        else:
            log_decision(
                "patch",
                "all",
                "rejeitado",
                model,
                "nao",
                remember=ui.remember_choice,
                expires_at=getattr(ui, "remember_expires", None),
            )
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
    "decisoes": handle_decisoes,
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
    "aprovar_proxima": handle_aprovar_proxima,
    "aprovar_durante": handle_aprovar_durante,
    "modo": handle_modo,
    "regras": handle_regras,
    "sugerir_regras": handle_sugerir_regras,
    "resetar": handle_resetar,
    "tests_local": handle_tests_local,
}
