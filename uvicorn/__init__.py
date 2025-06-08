import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

print("\u26A0\uFE0F Dependência ausente: uvicorn. Reverter para stub temporário.")


class Config:
    def __init__(self, app, host="127.0.0.1", port=8000, log_level="info"):
        self.app = app
        self.host = host
        self.port = port
        self.log_level = log_level

class Server:
    def __init__(self, config):
        self.config = config

    async def serve(self):
        app = self.config.app
        routes = getattr(app, "routes", {})

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in routes and "GET" in routes[self.path]:
                    resp = asyncio.run(routes[self.path]())
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(resp).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode()
                data = json.loads(body) if body else {}
                if self.path in routes and "POST" in routes[self.path]:
                    resp = asyncio.run(routes[self.path](**data))
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(resp).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        server = HTTPServer((self.config.host, self.config.port), Handler)
        print(f"Serving on {self.config.host}:{self.config.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


def run(app, host="127.0.0.1", port=8000, log_level="info"):
    cfg = Config(app, host=host, port=port, log_level=log_level)
    asyncio.run(Server(cfg).serve())
