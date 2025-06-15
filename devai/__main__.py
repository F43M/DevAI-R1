import asyncio
import argparse
import json
import difflib
import tempfile
import subprocess
import hashlib
import shutil
from pathlib import Path

from .config import config
from .core import CodeMemoryAI
from .cli import cli_main
from .dependency_check import check_dependencies


def main():
    parser = argparse.ArgumentParser(
        description="CodeMemoryAI - Assistente de Código Inteligente"
    )
    parser.add_argument("--api", action="store_true", help="Inicia o servidor API")
    parser.add_argument(
        "--cli", action="store_true", help="Inicia a interface de linha de comando"
    )
    parser.add_argument("--tui", action="store_true", help="Inicia a interface textual")
    parser.add_argument(
        "--guided", action="store_true", help="Mostra orientações passo a passo"
    )
    parser.add_argument(
        "--plain", action="store_true", help="Interface simples sem Rich"
    )
    parser.add_argument(
        "--no-log", action="store_true", help="Não registra histórico de chat"
    )
    parser.add_argument(
        "--observer", action="store_true", help="Modo observador passivo"
    )
    parser.add_argument(
        "--approval-mode",
        choices=["suggest", "auto_edit", "full_auto"],
        default=config.APPROVAL_MODE,
        help="Como lidar com operações sensíveis",
    )
    parser.add_argument("command", nargs="*", help="Comandos adicionais")
    args = parser.parse_args()
    check_dependencies()
    config.APPROVAL_MODE = args.approval_mode
    if not config.OPENROUTER_API_KEY:
        print(
            "\u26d4\ufe0f Nenhuma chave OPENROUTER_API_KEY encontrada. Algumas funcionalidades podem não funcionar."
        )
    if args.api:

        async def start_api() -> None:
            ai = CodeMemoryAI()
            await ai.run()

        asyncio.run(start_api())
        return
    if args.observer:

        async def run_observer() -> None:
            ai = CodeMemoryAI()
            await ai._learning_loop()

        asyncio.run(run_observer())
        return
    if args.cli or args.tui:
        asyncio.run(
            cli_main(
                guided=args.guided,
                plain=args.plain,
                log=not args.no_log,
                tui=args.tui,
            )
        )
        return

    if args.command:
        cmd = args.command

        async def handle_command() -> None:
            ai = CodeMemoryAI()
            from .learning_engine import LearningEngine

            engine = LearningEngine(ai.analyzer, ai.memory, ai.ai_model)

            if cmd[0] == "aprender":
                sub = cmd[1] if len(cmd) > 1 else "auto"
                if sub == "auto":
                    await engine.learn_from_codebase()
                    await engine.learn_from_errors()
                    await engine.extract_positive_patterns()
                    await engine.reflect_on_internal_knowledge()
                elif sub == "erros":
                    await engine.learn_from_errors()
                elif sub == "positivos":
                    await engine.extract_positive_patterns()
                elif sub == "projeto" and len(cmd) > 2:
                    await engine.import_external_codebase(cmd[2])
                else:
                    print("Comando aprender invalido")
                return
            elif cmd[0] == "aprendizado" and len(cmd) > 1 and cmd[1] == "resumo":
                path = Path("logs/learning_report.md")
                if path.exists():
                    print(path.read_text())
                else:
                    print("Nenhum resumo disponivel")
                return
            elif cmd[0] == "treinamento" and len(cmd) > 1 and cmd[1] == "profundo":
                from .symbolic_training import run_symbolic_training

                result = await run_symbolic_training(
                    ai.analyzer, ai.memory, ai.ai_model
                )
                print(json.dumps(result, indent=2))
                return
            elif cmd[0] == "fine_tune" and len(cmd) > 2:
                from .rlhf import RLFineTuner

                tuner = RLFineTuner(ai.memory)
                result = await tuner.fine_tune(cmd[1], cmd[2])
                print(json.dumps(result, indent=2))
                return
            elif cmd[0] == "preferencia" and len(cmd) > 1:
                from .feedback import registrar_preferencia

                registrar_preferencia(" ".join(cmd[1:]))
                print("Preferência registrada com sucesso")
                return
            elif cmd[0] == "simular" and len(cmd) > 2:
                from .shadow_mode import (
                    simulate_update,
                    evaluate_change_with_ia,
                    log_simulation,
                )

                args = cmd[1:]
                save_patch = False
                side_by_side = False
                rollback = False
                if "--patch" in args:
                    save_patch = True
                    args.remove("--patch")
                if "--html" in args:
                    side_by_side = True
                    args.remove("--html")
                if "--rollback" in args:
                    rollback = True
                    args.remove("--rollback")
                if len(args) < 2:
                    print("Uso: simular <arquivo> <codigo> [--patch] [--html]")
                    return

                file_path = args[0]
                new_code = " ".join(args[1:])
                diff, tests_ok, test_out, sim_id, patch_path = simulate_update(
                    file_path, new_code
                )
                patch_hash = hashlib.sha256(diff.encode("utf-8")).hexdigest()
                evaluation = await evaluate_change_with_ia(diff)

                if side_by_side:
                    original = Path(file_path).read_text().splitlines()
                    updated = new_code.splitlines()
                    html = difflib.HtmlDiff().make_table(
                        original,
                        updated,
                        fromdesc="original",
                        todesc="sugerido",
                        context=True,
                    )
                    print(html)
                else:
                    print(diff)

                if save_patch:
                    print(f"Patch salvo em {patch_path}")

                print(evaluation["analysis"])
                if tests_ok and input("Aplicar? [s/N] ").lower() == "s":
                    from .update_manager import UpdateManager

                    success, _ = UpdateManager().safe_apply(
                        file_path, lambda p: p.write_text(new_code), capture_output=True
                    )
                    if not success and rollback:
                        # shell: rollback patch
                        git_path = shutil.which("git") or "git"
                        subprocess.run(
                            [git_path, "apply", "-R", patch_path],
                            cwd=config.CODE_ROOT,
                            check=True,
                        )
                    action = "shadow_approved" if success else "shadow_failed"
                else:
                    action = "shadow_declined" if tests_ok else "shadow_failed"
                log_simulation(
                    sim_id,
                    file_path,
                    tests_ok,
                    evaluation["analysis"],
                    action,
                    patch_hash=patch_hash,
                    test_output=test_out,
                )
                return
            elif cmd[0] == "monitorar":
                from .monitor_engine import auto_monitor_cycle

                result = await auto_monitor_cycle(ai.analyzer, ai.memory, ai.ai_model)
                print(json.dumps(result, indent=2))
                return

        asyncio.run(handle_command())
        return

    print("Por favor, especifique --api ou --cli para iniciar o aplicativo")


if __name__ == "__main__":
    main()
