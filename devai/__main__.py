import asyncio
import argparse
import json
from pathlib import Path

from .config import config
from .core import CodeMemoryAI
from .cli import cli_main
from .dependency_check import check_dependencies


def main():
    parser = argparse.ArgumentParser(description="CodeMemoryAI - Assistente de Código Inteligente")
    parser.add_argument("--api", action="store_true", help="Inicia o servidor API")
    parser.add_argument("--cli", action="store_true", help="Inicia a interface de linha de comando")
    parser.add_argument("--observer", action="store_true", help="Modo observador passivo")
    parser.add_argument("command", nargs="*", help="Comandos adicionais")
    args = parser.parse_args()
    check_dependencies()
    if not config.OPENROUTER_API_KEY:
        print("Erro: A variável de ambiente OPENROUTER_API_KEY não está definida")
        return
    if args.api:
        ai = CodeMemoryAI()
        asyncio.run(ai.run())
        return
    if args.observer:
        ai = CodeMemoryAI()
        asyncio.run(ai._learning_loop())
        return
    if args.cli:
        asyncio.run(cli_main())
        return

    if args.command:
        cmd = args.command
        ai = CodeMemoryAI()
        from .learning_engine import LearningEngine
        engine = LearningEngine(ai.analyzer, ai.memory, ai.ai_model)

        if cmd[0] == "aprender":
            sub = cmd[1] if len(cmd) > 1 else "auto"
            if sub == "auto":
                asyncio.run(engine.learn_from_codebase())
                asyncio.run(engine.learn_from_errors())
                asyncio.run(engine.extract_positive_patterns())
                asyncio.run(engine.reflect_on_internal_knowledge())
            elif sub == "erros":
                asyncio.run(engine.learn_from_errors())
            elif sub == "positivos":
                asyncio.run(engine.extract_positive_patterns())
            elif sub == "projeto" and len(cmd) > 2:
                asyncio.run(engine.import_external_codebase(cmd[2]))
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
            result = asyncio.run(run_symbolic_training(ai.analyzer, ai.memory, ai.ai_model))
            print(json.dumps(result, indent=2))
            return
        elif cmd[0] == "preferencia" and len(cmd) > 1:
            from .feedback import registrar_preferencia
            registrar_preferencia(" ".join(cmd[1:]))
            print("Preferência registrada com sucesso")
            return
        elif cmd[0] == "simular" and len(cmd) > 2:
            from .shadow_mode import simulate_update, evaluate_change_with_ia, log_simulation, run_tests_in_temp
            file_path = cmd[1]
            new_code = " ".join(cmd[2:])
            diff, temp_root, sim_id = simulate_update(file_path, new_code)
            tests_ok, _ = run_tests_in_temp(temp_root)
            evaluation = asyncio.run(evaluate_change_with_ia(diff))
            print(diff)
            print(evaluation["analysis"])
            if tests_ok and input("Aplicar? [s/N] ").lower() == "s":
                from .update_manager import UpdateManager
                UpdateManager().safe_apply(file_path, lambda p: p.write_text(new_code))
                action = "shadow_approved"
            else:
                action = "shadow_declined" if tests_ok else "shadow_failed"
            log_simulation(file_path, evaluation["analysis"], action)
            return
        elif cmd[0] == "monitorar":
            from .monitor_engine import auto_monitor_cycle
            result = asyncio.run(auto_monitor_cycle(ai.analyzer, ai.memory, ai.ai_model))
            print(json.dumps(result, indent=2))
            return

    print("Por favor, especifique --api ou --cli para iniciar o aplicativo")


if __name__ == "__main__":
    main()
