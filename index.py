"""Vercel entrypoint: the web app, which only queues clips (`worker.convert` converts them)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from web.app import create_app  # noqa: E402

app = create_app()
