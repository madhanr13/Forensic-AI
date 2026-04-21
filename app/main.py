"""
ForensicAI — FastAPI Application Entry Point

Serves the web dashboard and exposes REST API endpoints for
image forensic analysis.
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import WEB_DIR, UPLOAD_DIR, RESULTS_DIR
from app.routes.analysis import router as analysis_router

# ────────────────────────────────────────────────────────────────────────
# App
# ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ForensicAI",
    description="Intelligent Digital Image Forensics & Manipulation Detection Platform",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/css", StaticFiles(directory=str(WEB_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(WEB_DIR / "js")), name="js")
app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")

# API routes
app.include_router(analysis_router, prefix="/api")


# ────────────────────────────────────────────────────────────────────────
# Root — serve dashboard
# ────────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ForensicAI",
        "version": "1.0.0",
    }
