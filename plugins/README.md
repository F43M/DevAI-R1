# Plugins

Coloque módulos Python neste diretório para estender o sistema de tarefas.
Cada plugin deve definir uma função `register(task_manager)` que recebe a
instância de `TaskManager` e adiciona novas tarefas ou métodos.

Exemplo simples:
```python
# plugins/todo_counter.py
from pathlib import Path

def register(tm):
    async def _perform_todo_counter_task(self, task, *args):
        count = 0
        for path in Path(self.code_analyzer.code_root).rglob('*.py'):
            for line in path.read_text().splitlines():
                if 'TODO' in line:
                    count += 1
        return [f'TODOs encontrados: {count}']

    tm.tasks['todo_counter'] = {
        'name': 'Contar TODOs',
        'type': 'todo_counter',
        'description': 'Conta marcações TODO no código',
    }
    setattr(tm, '_perform_todo_counter_task', _perform_todo_counter_task.__get__(tm))
```
