
from __future__ import annotations

import importlib.util
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import generate_answer as rag_generate_answer

FINETUNE_MODULE_PATH = os.path.join("finetune", "inference.py")

app = FastAPI(title="Smart Study Assistant API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_finetune_module = None
_finetune_error: Optional[str] = None


def _load_finetune_module() -> bool:
    """Attempt to load the local finetune inference module lazily."""
    global _finetune_module, _finetune_error

    if _finetune_module is not None:
        return True

    if not os.path.exists(FINETUNE_MODULE_PATH):
        _finetune_error = f"Finetune module not found at {FINETUNE_MODULE_PATH}"
        return False

    try:
        spec = importlib.util.spec_from_file_location(
            "finetune_inference", FINETUNE_MODULE_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to create module spec for finetune inference.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "generate_answer"):
            raise AttributeError(
                "Finetune module must expose generate_answer(subject, query)."
            )

        _finetune_module = module
        _finetune_error = None
        return True
    except Exception as exc:  
        _finetune_error = str(exc)
        _finetune_module = None
        return False


class QueryPayload(BaseModel):
    subject: str = Field(..., description="Subject key matching vector database name.")
    query: str = Field(..., description="User question text.")
    engine: str = Field(
        "rag",
        description="rag | finetune | auto (prefer finetune, fallback to rag)",
        pattern="^(?i)(rag|finetune|auto)$",
    )
    follow_up: bool = Field(
        False, description="Indicates whether this is a follow-up question."
    )
    last_query: Optional[str] = Field(
        None, description="Original query to use for follow-up messaging."
    )


@app.get("/")
async def health_check():
    """Simple health endpoint."""
    return {
        "status": "ok",
        "finetune_loaded": _finetune_module is not None,
        "finetune_error": _finetune_error,
    }


@app.post("/ask")
async def ask_question(payload: QueryPayload):
    """Proxy endpoint that routes requests to RAG or finetune engines."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    engine = payload.engine.lower()
    backend_used = ""
    answer = ""
    accuracy = 0.0

    def run_rag():
        nonlocal answer, accuracy, backend_used
        answer, accuracy = rag_generate_answer(payload.subject, query)
        backend_used = "rag"

    def run_finetune():
        nonlocal answer, accuracy, backend_used
        if _finetune_module is None and not _load_finetune_module():
            raise HTTPException(
                status_code=503,
                detail=_finetune_error
                or "Finetune backend unavailable. See server logs for details.",
            )
        try:
            result = _finetune_module.generate_answer(payload.subject, query)
            if isinstance(result, tuple):
                answer, accuracy = result
            else:
                answer, accuracy = result, 0.0
            backend_used = "finetune"
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if engine == "finetune":
        run_finetune()
    elif engine == "rag":
        run_rag()
    else:  # auto mode
        if _load_finetune_module():
            try:
                run_finetune()
            except HTTPException:
                run_rag()
        else:
            run_rag()

    return {
        "answer": answer,
        "accuracy": accuracy or 0.0,
        "backend": backend_used,
        "echo_query": query,
        "enable_followup": True,
    }

