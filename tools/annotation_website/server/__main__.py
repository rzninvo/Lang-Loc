"""Run as ``python -m server`` from the annotation_website directory."""
from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    s = get_settings()
    uvicorn.run("server.main:app", host=s.host, port=s.port, reload=s.reload)


if __name__ == "__main__":
    main()
