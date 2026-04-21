"""
ForensicAI — One-Command Launcher
Run this to start the application: python run.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Create required directories
for d in ["data/uploads", "data/results", "data/models", "data/sample_images", "web/assets"]:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)


def main():
    import uvicorn
    from app.config import HOST, PORT, DEBUG

    print(f"    🌐 Starting server at http://{HOST}:{PORT}")
    print(f"    📊 Dashboard:       http://{HOST}:{PORT}/")
    print(f"    📋 API Docs:        http://{HOST}:{PORT}/docs")
    print(f"    🔧 Debug mode:      {DEBUG}")
    print(f"    {'─' * 50}")
    print()

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info",
    )


if __name__ == "__main__":
    main()
