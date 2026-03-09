"""
SAT Tutor - Web UI (step-by-step, same flow as CLI)

Run: python -m sat_tutor.web_app
Then open http://127.0.0.1:8000 in browser.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core.pipeline import GREMathPipeline
from .core.diagnose import ErrorDiagnoser
from .io.json_io import create_session_output, save_session_result
from .io.report_md import save_report_md

app = FastAPI(title="SAT Tutor Web", description="Step-by-step same as CLI")

# In-memory session store: session_id -> { "pipeline": pipeline, "step": ... }
sessions: dict = {}
OUTPUT_DIR = os.environ.get("SAT_TUTOR_OUTPUT_DIR", "outputs")
# Project root for listing PDFs: env SAT_TUTOR_BASE_DIR, or cwd, or package parent (sat-math-tutor)
_cwd = Path(os.getcwd()).resolve()
_pkg_dir = Path(__file__).resolve().parent  # sat_tutor/
BASE_DIR = Path(os.environ.get("SAT_TUTOR_BASE_DIR", _cwd)).resolve()
if not (BASE_DIR / "data").is_dir() and (_pkg_dir.parent / "data").is_dir():
    BASE_DIR = _pkg_dir.parent
# Allowed directory for PDF selection (relative to BASE_DIR)
PDF_ROOT = "data"


# --- Request/Response models ---

class StartForm(BaseModel):
    subject: str = "math"
    mode: str = "diagnose"
    pages: str = "all"
    dpi: int = 300


class CorrectAnswersChoice(BaseModel):
    use_llm: bool = True  # True = use LLM to solve, False = use uploaded file


class UserAnswersChoice(BaseModel):
    method: str  # "file" | "interactive" | "simulated"
    answers: Optional[dict] = None  # for interactive: { "p1_q1": "A", ... }


class DiagnoseOptions(BaseModel):
    mode: str = "B"  # A, B, C
    feedback_timing: str = "after_all"


# --- API ---

def _list_project_pdfs() -> list[str]:
    """List PDF paths under BASE_DIR/PDF_ROOT, relative to BASE_DIR."""
    root = BASE_DIR / PDF_ROOT
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*.pdf"):
        try:
            rel = p.relative_to(BASE_DIR)
            out.append(str(rel).replace("\\", "/"))
        except ValueError:
            continue
    return sorted(out)


def _resolve_pdf_path(relative_path: str) -> Path:
    """Resolve and validate: must be under BASE_DIR/PDF_ROOT and exist."""
    # Avoid path traversal
    clean = Path(relative_path).as_posix()
    if clean.startswith("/") or ".." in clean:
        raise HTTPException(status_code=400, detail="Invalid path")
    full = (BASE_DIR / clean).resolve()
    root = (BASE_DIR / PDF_ROOT).resolve()
    if not str(full).startswith(str(root)) or not full.is_file():
        raise HTTPException(status_code=400, detail="PDF not found or not under project data")
    return full


@app.get("/api/pdfs")
async def api_list_pdfs():
    """List PDF files in project data directory (e.g. data/samples/...)."""
    return {"paths": _list_project_pdfs()}


@app.post("/api/sessions/start")
async def api_sessions_start(
    pdf_path: str = Form(..., description="Relative path from project root, e.g. data/samples/Linear_Equations.pdf"),
    subject: str = Form("math"),
    mode: str = Form("diagnose"),
    pages: str = Form("all"),
    dpi: int = Form(300),
):
    """Step 1: Select PDF from project, run Stage 0 + T (transcribe). Returns session_id and question count."""
    path = _resolve_pdf_path(pdf_path)
    try:
        pipeline = GREMathPipeline(use_mock=False, output_dir=OUTPUT_DIR, subject=subject)
        result = pipeline.run_transcribe_step(
            pdf_path=str(path),
            mode=mode,
            pages=pages,
            dpi=dpi,
            transcribed_json=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    session_id = result["session_id"]
    sessions[session_id] = {
        "pipeline": pipeline,
        "step": "transcribe_done",
    }
    return {
        "session_id": session_id,
        "question_count": result["question_count"],
        "failed_pages": result.get("failed_pages", []),
        "errors": result.get("errors", []),
    }


@app.get("/api/sessions/{session_id}/questions")
async def api_sessions_questions(session_id: str):
    """Return question list for interactive answer input (id, stem, choices, problem_type)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    rec = sessions[session_id]
    if rec["step"] not in ("transcribe_done", "solve_done", "answers_done", "diagnose_done"):
        raise HTTPException(status_code=400, detail="Invalid step")
    pipeline = rec["pipeline"]
    questions = pipeline._web_questions
    out = []
    for q in questions:
        out.append({
            "id": q.id,
            "stem": q.stem,
            "choices": q.choices,
            "problem_type": q.problem_type,
            "latex_equations": getattr(q, "latex_equations", []) or [],
            "diagram_description": getattr(q, "diagram_description", None),
        })
    return {"questions": out}


@app.post("/api/sessions/{session_id}/correct-answers")
async def api_sessions_correct_answers(
    session_id: str,
    use_llm: bool = Form(True),
    file: Optional[UploadFile] = File(None),
):
    """
    Step 2: Choose correct answers source.
    - use_llm=true: use LLM to solve (no file).
    - use_llm=false: upload correct answers JSON file.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    rec = sessions[session_id]
    if rec["step"] == "solve_done":
        # Idempotent: already did Step 2 (e.g. previous request succeeded but client didn't get response)
        pipeline = rec["pipeline"]
        return {"solve_count": len(pipeline._web_solve_results), "errors": [], "already_done": True}
    if rec["step"] != "transcribe_done":
        raise HTTPException(
            status_code=400,
            detail=f"Session is at step '{rec['step']}', but this action requires transcribe_done. Try refreshing and continuing from the next step."
        )
    pipeline = rec["pipeline"]
    correct_path = None
    if not use_llm and file and file.filename:
        suffix = Path(file.filename).suffix or ".json"
        path = os.path.join(pipeline.session_dir, "correct_answers" + suffix)
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
        correct_path = path
    try:
        result = pipeline.run_solve_step(
            use_correct_answers_file=bool(correct_path),
            correct_answers_path=correct_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    rec["step"] = "solve_done"
    return {"solve_count": result["solve_count"], "errors": result.get("errors", [])}


@app.post("/api/sessions/{session_id}/user-answers")
async def api_sessions_user_answers(
    session_id: str,
    method: str = Form(...),  # file | interactive | simulated
    file: Optional[UploadFile] = File(None),
    answers: Optional[str] = Form(None),  # JSON string for interactive
    feedback_timing: Optional[str] = Form(None),  # per_question | after_all (for interactive)
):
    """
    Step 3: Provide user answers.
    - method=file: upload user answers JSON file.
    - method=interactive + feedback_timing=per_question: start one-by-one flow (no answers yet).
    - method=interactive + answers=...: submit all answers at once (after_all).
    - method=simulated: run AI student simulator (no file).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    rec = sessions[session_id]
    if rec["step"] != "solve_done":
        raise HTTPException(status_code=400, detail="Expected step: solve_done")
    pipeline = rec["pipeline"]
    if method == "interactive" and feedback_timing == "per_question":
        rec["per_question"] = True
        rec["diagnose_results"] = []
        pipeline._web_user_answers = {}
        pipeline._web_answer_input_meta = {"input_mode": "interactive", "feedback_timing": "per_question"}
        pipeline.set_diagnose_options_step(mode="B", feedback_timing="per_question")
        return {"ok": True, "per_question": True}
    file_path = None
    answers_dict = None
    if method == "file" and file and file.filename:
        path = os.path.join(pipeline.session_dir, "user_answers.json")
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
        file_path = path
    elif method == "interactive" and answers is not None:
        import json
        try:
            answers_dict = json.loads(answers) if answers else {}
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    try:
        pipeline.set_user_answers_step(
            method="file" if method == "file" and file_path else ("simulated" if method == "simulated" else "interactive"),
            file_path=file_path,
            answers_dict=answers_dict,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    rec["step"] = "answers_done"
    return {"ok": True}


@app.post("/api/sessions/{session_id}/answer-and-diagnose-one")
async def api_sessions_answer_and_diagnose_one(
    session_id: str,
    question_id: str = Form(...),
    answer: str = Form(...),
):
    """
    Per-question flow: submit one answer, get diagnosis for that question immediately.
    Allowed when session is in per_question mode (step remains solve_done).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    rec = sessions[session_id]
    if not rec.get("per_question"):
        raise HTTPException(status_code=400, detail="This session is not in per-question mode")
    if rec["step"] not in ("solve_done", "diagnose_done"):
        raise HTTPException(status_code=400, detail="Invalid step for per-question submit")
    pipeline = rec["pipeline"]
    questions = pipeline._web_questions
    solve_results = pipeline._web_solve_results
    solve_map = {sr.question_id: sr for sr in solve_results}
    question = next((q for q in questions if q.id == question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
    solve_result = solve_map.get(question_id)
    if not solve_result:
        raise HTTPException(status_code=400, detail=f"No solve result for {question_id}")
    pipeline._web_user_answers[question_id] = answer.strip()
    diagnoser = ErrorDiagnoser(pipeline.llm, pipeline.logger, subject=pipeline.subject)
    diagnose_mode = pipeline._web_diagnose_mode
    if diagnose_mode == "A":
        result, error = diagnoser.diagnose_mode_a(question, solve_result, answer)
    else:
        result, error = diagnoser.diagnose(question, solve_result, answer)
    if error and not result:
        raise HTTPException(status_code=500, detail=error)
    if result:
        rec.setdefault("diagnose_results", []).append(result)
    user_answers = pipeline._web_user_answers
    total = len(questions)
    answered = len(user_answers)
    all_done = answered >= total
    if all_done:
        rec["step"] = "diagnose_done"
        diagnose_results = rec.get("diagnose_results", [])
        session_result = create_session_output(
            session_id=pipeline.session_id,
            pdf_path=pipeline._web_pdf_path,
            mode=pipeline._web_mode,
            questions=questions,
            failed_pages=pipeline._web_failed_pages,
            errors=pipeline._web_errors,
            solve_results=solve_results,
            diagnose_results=diagnose_results,
            user_answers=user_answers,
        )
        results_path = os.path.join(pipeline.session_dir, "results.json")
        save_session_result(session_result, results_path)
        report_path = os.path.join(pipeline.session_dir, "report.md")
        save_report_md(session_result, report_path)
        rec["last_session_result"] = session_result
    return {
        "diagnose_result": result.model_dump() if result else None,
        "all_done": all_done,
        "answered": answered,
        "total": total,
    }


@app.post("/api/sessions/{session_id}/diagnose-options")
async def api_sessions_diagnose_options(
    session_id: str,
    body: DiagnoseOptions,
):
    """Step 4: Set diagnosis mode (A/B/C) and feedback timing. Allowed after solve_done (choose mode before answering)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    rec = sessions[session_id]
    allowed = ("solve_done", "answers_done")
    if rec["step"] not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Expected step: solve_done or answers_done (current: {rec['step']}). Complete Step 2 (Run solve) first.",
        )
    rec["pipeline"].set_diagnose_options_step(
        mode=body.mode or "B",
        feedback_timing=body.feedback_timing or "after_all",
    )
    return {"ok": True}


@app.post("/api/sessions/{session_id}/run-diagnose")
async def api_sessions_run_diagnose(session_id: str):
    """Step 5: Run diagnosis, return summary and report path. Idempotent if already diagnose_done (e.g. per-question flow)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    rec = sessions[session_id]
    if rec["step"] == "diagnose_done":
        pipeline = rec["pipeline"]
        result = rec.get("last_session_result")
        if not result:
            from .io.json_io import load_session_result
            rpath = os.path.join(pipeline.session_dir, "results.json")
            if os.path.isfile(rpath):
                result = load_session_result(rpath)
        if result:
            return {
                "session_id": result.session_id,
                "total_questions": result.total_questions,
                "answered_questions": result.answered_questions,
                "correct_count": result.correct_count,
                "incorrect_ids": result.incorrect_ids or [],
                "session_dir": pipeline.session_dir,
                "report_path": os.path.join(pipeline.session_dir, "report.md"),
            }
    if rec["step"] != "answers_done":
        raise HTTPException(status_code=400, detail="Expected step: answers_done")
    try:
        result = rec["pipeline"].run_diagnose_step()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    rec["step"] = "diagnose_done"
    rec["last_session_result"] = result
    report_path = os.path.join(rec["pipeline"].session_dir, "report.md")
    return {
        "session_id": result.session_id,
        "total_questions": result.total_questions,
        "answered_questions": result.answered_questions,
        "correct_count": result.correct_count,
        "incorrect_ids": result.incorrect_ids,
        "session_dir": rec["pipeline"].session_dir,
        "report_path": report_path,
    }


@app.get("/api/sessions/{session_id}/report")
async def api_sessions_report(session_id: str):
    """Return report.md content."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    report_path = os.path.join(sessions[session_id]["pipeline"].session_dir, "report.md")
    if not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="Report not generated yet")
    with open(report_path, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read())


@app.get("/api/sessions/{session_id}/logs")
async def api_sessions_logs(session_id: str):
    """Return session log file content if present."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    log_path = os.path.join(sessions[session_id]["pipeline"].session_dir, "logs.txt")
    if not os.path.isfile(log_path):
        return PlainTextResponse("")
    with open(log_path, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read())


# --- Serve frontend ---

static_dir = Path(__file__).parent / "web" / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html = Path(__file__).parent / "web" / "index.html"
    if not html.is_file():
        return HTMLResponse(
            "<h1>SAT Tutor Web</h1><p>Missing web/index.html. Create sat_tutor/web/index.html.</p>"
        )
    return HTMLResponse(html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    # 127.0.0.1 可在本机浏览器直接打开；若需局域网访问再改为 0.0.0.0
    uvicorn.run(app, host="127.0.0.1", port=8000)
