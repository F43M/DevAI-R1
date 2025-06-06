from pathlib import Path


def register(task_manager):
    async def _perform_todo_counter_task(self, task, *args):
        count = 0
        for path in Path(self.code_analyzer.code_root).rglob('*.py'):
            try:
                for line in path.read_text().splitlines():
                    if 'TODO' in line:
                        count += 1
            except Exception:
                continue
        return [f'TODOs encontrados: {count}']

    task_manager.tasks['todo_counter'] = {
        'name': 'Contar TODOs',
        'type': 'todo_counter',
        'description': 'Conta marcações TODO no código',
    }
    setattr(task_manager, '_perform_todo_counter_task', _perform_todo_counter_task.__get__(task_manager))
