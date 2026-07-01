"""Serve the static chat UI from the same app as the API, so the browser hits one origin.

The UI is one self-contained ``static/index.html`` (no build step, no framework). Registering
it as an explicit ``GET /`` route — rather than mounting ``StaticFiles`` at ``/`` — keeps it
from shadowing the ``/v1/*`` API routes, and means the engine speaks both the OpenAI wire
format and its own front end without a separate server or any CORS handling.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

_INDEX = Path(__file__).parent / "static" / "index.html"


def register_webui(app: FastAPI) -> None:
    """Add a ``GET /`` route serving the bundled chat UI onto an existing app."""

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_INDEX, media_type="text/html")
