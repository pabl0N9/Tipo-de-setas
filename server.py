import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from api.predict import API_VERSION, predict_one

ROOT_DIR = Path(__file__).parent
PUBLIC_DIR = ROOT_DIR / "public"


class AppHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Silence default logging to keep Render logs cleaner.
        return

    # --- API helpers ---
    def _set_api_headers(self, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_OPTIONS(self):
        if self.path.startswith("/api/predict"):
            self._set_api_headers(200)
        else:
            super().do_OPTIONS()

    # --- API endpoints ---
    def handle_api_predict_get(self):
        body = {
            "status": "ok",
            "message": "POST to this endpoint with mushroom features.",
            "version": API_VERSION,
        }
        self._set_api_headers(200)
        self.wfile.write(json.dumps(body).encode())

    def handle_api_predict_post(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length else b""

        try:
            payload = json.loads(raw_body) if raw_body else {}
            result = predict_one(payload)
            response = {"status": "success", "version": API_VERSION, "prediction": result}
            self._set_api_headers(200)
            self.wfile.write(json.dumps(response).encode())
        except Exception as exc:  # noqa: BLE001
            error_message = {"status": "error", "version": API_VERSION, "message": str(exc)}
            self._set_api_headers(400)
            self.wfile.write(json.dumps(error_message).encode())

    # --- Static files ---
    def translate_path(self, path: str) -> str:
        """
        Map the requested path to the public directory.
        """
        parsed_path = urlparse(path).path
        rel_path = parsed_path.lstrip("/")

        # Serve index.html for root or directory paths
        if rel_path == "" or rel_path.endswith("/"):
            rel_path = rel_path + "index.html"

        return str(PUBLIC_DIR / rel_path)

    def do_GET(self):
        if self.path.startswith("/api/predict"):
            return self.handle_api_predict_get()
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/predict"):
            return self.handle_api_predict_post()
        self.send_error(404, "Not Found")


def run():
    port = int(os.environ.get("PORT", "10000"))
    httpd = HTTPServer(("", port), AppHandler)
    print(f"Serving on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
