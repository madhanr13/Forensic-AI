"""
ForensicAI — Analysis API Routes

Endpoints for running forensic analysis modules individually
or as a comprehensive suite.
"""

import uuid
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.config import UPLOAD_DIR
from app.services.orchestrator import ForensicOrchestrator

router = APIRouter(tags=["analysis"])

# Lazy-initialized orchestrator (loads models once)
_orchestrator: Optional[ForensicOrchestrator] = None


def get_orchestrator() -> ForensicOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ForensicOrchestrator()
    return _orchestrator


async def _save_upload(file: UploadFile) -> Path:
    """Save uploaded file and return path."""
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / filename

    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum 25MB.")

    with open(save_path, "wb") as f:
        f.write(content)

    return save_path


# ────────────────────────────────────────────────────────────────────────
# Full analysis
# ────────────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_full(file: UploadFile = File(...)):
    """Run all 6 forensic analysis modules on the uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    save_path = await _save_upload(file)

    try:
        orchestrator = get_orchestrator()
        start = time.time()
        results = orchestrator.analyze_all(str(save_path))
        elapsed = time.time() - start

        return JSONResponse({
            "success": True,
            "analysis_id": uuid.uuid4().hex,
            "filename": file.filename,
            "elapsed_seconds": round(elapsed, 2),
            "overall_score": results["overall_score"],
            "overall_verdict": results["overall_verdict"],
            "modules": results["modules"],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Cleanup uploaded file
        if save_path.exists():
            save_path.unlink()


# ────────────────────────────────────────────────────────────────────────
# Individual module endpoints
# ────────────────────────────────────────────────────────────────────────

@router.post("/analyze/ela")
async def analyze_ela(file: UploadFile = File(...)):
    """Run Error Level Analysis only."""
    return await _run_single_module(file, "ela")


@router.post("/analyze/copymove")
async def analyze_copymove(file: UploadFile = File(...)):
    """Run Copy-Move Forgery Detection only."""
    return await _run_single_module(file, "copymove")


@router.post("/analyze/noise")
async def analyze_noise(file: UploadFile = File(...)):
    """Run Noise Pattern Analysis only."""
    return await _run_single_module(file, "noise")


@router.post("/analyze/ai-detect")
async def analyze_ai_detection(file: UploadFile = File(...)):
    """Run AI-Generated Image Detection only."""
    return await _run_single_module(file, "ai_detection")


@router.post("/analyze/heatmap")
async def analyze_heatmap(file: UploadFile = File(...)):
    """Run Tampering Heatmap Generation only."""
    return await _run_single_module(file, "heatmap")


@router.post("/analyze/metadata")
async def analyze_metadata(file: UploadFile = File(...)):
    """Run Metadata Forensics only."""
    return await _run_single_module(file, "metadata")


async def _run_single_module(file: UploadFile, module_name: str):
    """Helper to run a single forensic module."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    save_path = await _save_upload(file)

    try:
        orchestrator = get_orchestrator()
        start = time.time()
        result = orchestrator.analyze_single(str(save_path), module_name)
        elapsed = time.time() - start

        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "elapsed_seconds": round(elapsed, 2),
            "result": result,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if save_path.exists():
            save_path.unlink()
