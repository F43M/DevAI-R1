import asyncio
import argparse

from .config import config
from .core import CodeMemoryAI
from .cli import cli_main


def main():
    parser = argparse.ArgumentParser(description="CodeMemoryAI - Assistente de Código Inteligente")
    parser.add_argument("--api", action="store_true", help="Inicia o servidor API")
    parser.add_argument("--cli", action="store_true", help="Inicia a interface de linha de comando")
    args = parser.parse_args()
    if not config.OPENROUTER_API_KEY:
        print("Erro: A variável de ambiente OPENROUTER_API_KEY não está definida")
        return
    if args.api:
        ai = CodeMemoryAI()
        asyncio.run(ai.run())
    elif args.cli:
        asyncio.run(cli_main())
    else:
        print("Por favor, especifique --api ou --cli para iniciar o aplicativo")


if __name__ == "__main__":
    main()
