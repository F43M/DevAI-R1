class ClientSession:
    async def post(self, *a, **k):
        class Response:
            status = 200
            async def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}
            async def text(self):
                return ""
        return Response()
    async def close(self):
        pass

class ClientTimeout:
    def __init__(self, total=None):
        pass
