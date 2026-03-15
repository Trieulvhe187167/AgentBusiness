"""
Run the local mock business API for end-to-end demos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


if __name__ == "__main__":
    uvicorn.run("app.mock_business_api:app", host="127.0.0.1", port=9001, reload=False)
