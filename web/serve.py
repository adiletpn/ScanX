"""HTTPS-сервер для телефонной версии защитника.

Браузеры дают доступ к микрофону только в защищённом контексте: HTTPS
или localhost. По обычному HTTP свойство ``navigator.mediaDevices``
не существует вовсе — страница молча остаётся без микрофона.

Сертификат самоподписанный, поэтому браузер покажет предупреждение.
Шифрование при этом настоящее; не хватает только подписи центра
сертификации, а для локальной сети она и не нужна.

Запуск:
    python web/serve.py
"""
import http.server
import ssl
from pathlib import Path

HERE = Path(__file__).parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def end_headers(self):
        # Модели весят десятки мегабайт — пусть браузер их кэширует.
        self.send_header("Cache-Control", "public, max-age=86400")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}", flush=True)


ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(HERE / "cert.pem", HERE / "key.pem")

server = http.server.ThreadingHTTPServer(("0.0.0.0", 8443), Handler)
server.socket = ctx.wrap_socket(server.socket, server_side=True)
print("HTTPS на порту 8443", flush=True)
server.serve_forever()
