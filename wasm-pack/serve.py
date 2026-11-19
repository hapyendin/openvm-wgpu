#!/usr/bin/env python3
# Static server + POST /log sink + /shutdown. Usage: serve.py <dir> <port> <logfile>
import sys, os, threading, http.server, functools

d, port, logfile = sys.argv[1], int(sys.argv[2]), sys.argv[3]

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=d, **kw)
    def log_message(self, *a):
        pass
    def _sink(self, body):
        with open(logfile, 'a') as f:
            f.write(body)
            f.flush()
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        self._sink(self.rfile.read(n).decode('utf-8', 'replace'))
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        if self.path == '/shutdown':
            threading.Timer(0.3, lambda: os._exit(0)).start()
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

http.server.ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
