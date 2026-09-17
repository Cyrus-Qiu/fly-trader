from __future__ import annotations

import json
import math
import threading
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_json_safe(item) for item in value]
    return value


class DashboardState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.paused = False
        self.latest: dict = {"status": "starting", "paused": False}
        self.assets: dict[str, dict] = {}
        self.accounts: dict[str, dict] = {}
        self.proposals: dict[str, dict] = {}
        self.performance: dict[str, dict] = {}
        self.series: deque[dict] = deque(maxlen=300)

    def update(self, latest: dict, append: bool = False) -> None:
        with self._lock:
            self.latest.update(latest)
            self.latest["paused"] = self.paused
            if append:
                self.series.append({
                    key: self.latest.get(key) for key in
                    ("elapsed_s", "price", "raw_score", "smooth_score", "stable_action")
                })

    def snapshot(self) -> dict:
        with self._lock:
            return _json_safe({
                "latest": dict(self.latest), "assets": dict(self.assets),
                "accounts": dict(self.accounts), "proposals": dict(self.proposals),
                "performance": dict(self.performance),
                "series": list(self.series),
            })

    def update_account(self, market: str, values: dict) -> None:
        with self._lock:
            current = self.accounts.setdefault(market, {"market": market})
            current.update(values)

    def get_account(self, market: str) -> dict | None:
        with self._lock:
            account = self.accounts.get(market)
            return dict(account) if account else None

    def update_proposal(self, symbol: str, values: dict) -> None:
        with self._lock:
            self.proposals[symbol] = dict(values)

    def update_performance(self, symbol: str, values: dict) -> None:
        with self._lock:
            self.performance[symbol] = dict(values)

    def update_asset(self, symbol: str, values: dict, selected: bool = False,
                     append: bool = False) -> None:
        with self._lock:
            current = self.assets.setdefault(symbol, {"symbol": symbol})
            current.update(values)
            if selected:
                self.latest.update(current)
                self.latest["paused"] = self.paused
            if append:
                point = {key: current.get(key) for key in
                         ("elapsed_s", "price", "raw_score", "smooth_score", "stable_action")}
                point["symbol"] = symbol
                self.series.append(point)

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.paused = paused
            self.latest["paused"] = paused

    def is_paused(self) -> bool:
        with self._lock:
            return self.paused


def start_dashboard(state: DashboardState, port: int = 8787,
                    open_browser: bool = True) -> ThreadingHTTPServer:
    web_root = Path(__file__).with_name("web")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/state":
                body = json.dumps(state.snapshot(), ensure_ascii=False).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
                return
            filename = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
            if filename not in {"index.html", "dashboard.css", "dashboard.js"}:
                self.send_error(404)
                return
            target = web_root / filename
            mime = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}[target.suffix]
            self._send(target.read_bytes(), f"{mime}; charset=utf-8")

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/pause":
                self.send_error(404)
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 1024)
                payload = json.loads(self.rfile.read(length) or b"{}")
                state.set_paused(bool(payload["paused"]))
                self._send(json.dumps({"paused": state.is_paused()}).encode(), "application/json")
            except (ValueError, KeyError, json.JSONDecodeError):
                self._send(b'{"error":"invalid request"}', "application/json", 400)

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(f"本地仪表盘：{url}")
    if open_browser:
        webbrowser.open(url)
    return server
