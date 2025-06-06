class Interpreter:
    def __call__(self, expression, **kwargs):
        try:
            return eval(expression, {}, kwargs)
        except Exception:
            return False
