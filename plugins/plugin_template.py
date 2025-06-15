"""Template simples para novos plugins.

Copie este arquivo para `plugins/<nome>.py` e ajuste as funções.
"""

# pylint: disable=no-value-for-parameter

PLUGIN_INFO = {
    "name": "Example Plugin",
    "version": "0.1",
    "description": "Descreva o que o plugin faz",
}


def register(task_manager):
    """Registrar tarefas ou métodos no TaskManager."""

    async def _perform_example(self, task, *args):
        # implemente o comportamento da tarefa
        return ["ok"]

    task_manager.tasks["example"] = {
        "name": "Exemplo",
        "type": "example",
        "description": "Explicação breve",
    }
    setattr(
        task_manager,
        "_perform_example",
        _perform_example.__get__(task_manager),
    )


def unregister(task_manager):
    """Remover tarefas e limpar atributos."""
    task_manager.tasks.pop("example", None)
    if hasattr(task_manager, "_perform_example"):
        delattr(task_manager, "_perform_example")
