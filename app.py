"""
Root ASGI entrypoint for deployment (Hugging Face Spaces / local).
Exposes `app` imported from backend.app.main.
Run locally: `uvicorn app:app --reload --port 8000`
"""
from __future__ import annotations

import os

from backend.app.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

