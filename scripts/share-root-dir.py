#!/usr/bin/env python3
"""Share the project root over HTTP on port 9000.

Usage:
    uv run python scripts/share-root-dir.py
    # or: run share-root-dir
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORT = 9000


def main() -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(PROJECT_ROOT))
    server = ThreadingHTTPServer(("0.0.0.0", PORT), handler)
    print(f"Sharing {PROJECT_ROOT} at http://localhost:{PORT}/")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nHTTP server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
