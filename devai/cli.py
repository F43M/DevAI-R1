import asyncio
import json

from .config import config, logger
from .error_handler import friendly_message, log_error
from .core import CodeMemoryAI
from .feedback import FeedbackDB, registrar_preferencia
from pathlib import Path

from .ui import CLIUI
from .update_manager import UpdateManager
from .command_router import COMMANDS, handle_default
from .decision_log import log_decision
import devai.command_router as command_router_module
command_router_module.UpdateManager = UpdateManager
try:
    from .tui import TUIApp
except Exception:  # pragma: no cover - optional dependency
    TUIApp = None  # type: ignore



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
    # Ensure router uses the potentially patched registrar_preferencia
    import devai.command_router as command_router_module
    command_router_module.registrar_preferencia = registrar_preferencia
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
    print("/modo <suggest|auto_edit|full_auto> - Altera modo de aprovação")
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
