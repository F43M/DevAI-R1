import asyncio

print("\u26A0\uFE0F Dependência ausente: aiohttp. Reverter para stub temporário.")


class Response:
    def __init__(self, text: str, status: int = 200):
        self.status = status
        self._text = text

    async def json(self):
        return {"choices": [{"message": {"content": self._text}}]}

    async def text(self):
        return self._text


class ClientSession:
    async def post(self, url, *, headers=None, json=None, timeout=None):
        prompt = ""
        if json and isinstance(json.get("messages"), list):
            prompt = json["messages"][-1].get("content", "")
        return Response(f"Resposta: {prompt}")

    async def close(self):
        await asyncio.sleep(0)


class ClientTimeout:
    def __init__(self, total=None):
        self.total = total
