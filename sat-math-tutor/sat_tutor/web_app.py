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
from .core.models import DiagnoseResult
from .io.json_io import create_session_output, save_session_result
from .io.report_md import save_report_md
from .io.answers import _transcribe_handwritten_work_image

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


def _serialize_question(question) -> dict:
    return {
        "id": question.id,
        "stem": question.stem,
        "choices": question.choices,
        "problem_type": question.problem_type,
        "latex_equations": getattr(question, "latex_equations", []) or [],
        "diagram_description": getattr(question, "diagram_description", None),
    }


def _completed_count(rec: dict) -> int:
    return len(rec.get("diagnose_results", []))


def _solve_map(pipeline: GREMathPipeline) -> dict:
    return {sr.question_id: sr for sr in pipeline._web_solve_results}


def _make_mode_c_correct_result(question, solve_result, answer: str, work_info: Optional[dict] = None) -> DiagnoseResult:
    correct_answer = solve_result.correct_answer.strip()
    result = DiagnoseResult(
        question_id=question.id,
        user_answer=answer.strip(),
        correct_answer=correct_answer,
        is_correct=True,
        why_user_choice_is_tempting=None,
        likely_misconceptions=[],
        how_to_get_correct=None,
        option_analysis=[],
    )
    if work_info:
        result.student_work_image_path = work_info.get("image_path")
        result.student_work_transcription = work_info.get("transcribed_work")
    return result


def _finalize_web_session(rec: dict) -> dict:
    pipeline = rec["pipeline"]
    session_result = create_session_output(
        session_id=pipeline.session_id,
        pdf_path=pipeline._web_pdf_path,
        mode=pipeline._web_mode,
        questions=pipeline._web_questions,
        failed_pages=pipeline._web_failed_pages,
        errors=pipeline._web_errors,
        solve_results=pipeline._web_solve_results,
        diagnose_results=rec.get("diagnose_results", []),
        user_answers=pipeline._web_user_answers,
    )
    results_path = os.path.join(pipeline.session_dir, "results.json")
    save_session_result(session_result, results_path)
    report_path = os.path.join(pipeline.session_dir, "report.md")
    save_report_md(session_result, report_path)
    rec["last_session_result"] = session_result
    rec["step"] = "diagnose_done"
    return {
        "session_id": session_result.session_id,
        "total_questions": session_result.total_questions,
        "answered_questions": session_result.answered_questions,
        "correct_count": session_result.correct_count,
        "incorrect_ids": session_result.incorrect_ids or [],
        "session_dir": pipeline.session_dir,
        "report_path": report_path,
    }


def _start_mode_c_question(
    rec: dict,
    question_id: str,
    *,
    first_attempt: Optional[str] = None,
    current_attempt: Optional[str] = None,
    attempts_used: int = 1,
) -> dict:
    pipeline = rec["pipeline"]
    question = next((q for q in pipeline._web_questions if q.id == question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
    solve_result = _solve_map(pipeline).get(question_id)
    if not solve_result:
        raise HTTPException(status_code=400, detail=f"No solve result for {question_id}")
    first_attempt = (first_attempt if first_attempt is not None else pipeline._web_user_answers.get(question_id) or "").strip()
    current_attempt = (current_attempt if current_attempt is not None else first_attempt).strip()
    student_work_text = (pipeline._web_student_work_map.get(question_id) or {}).get("transcribed_work") or None
    diagnoser = ErrorDiagnoser(pipeline.llm, pipeline.logger, subject=pipeline.subject)
    hint_result = diagnoser.get_hint_for_wrong_answer(
        question=question,
        solve_result=solve_result,
        user_answer=current_attempt,
        student_work_text=student_work_text,
    )
    rec["mode_c_pending"] = {
        "question_id": question_id,
        "first_attempt": first_attempt,
        "attempts_used": attempts_used,
    }
    return {
        "mode_c_pending": True,
        "question": _serialize_question(question),
        "first_attempt": first_attempt,
        "current_attempt": current_attempt,
        "hint_result": hint_result,
        "attempts_used": attempts_used,
        "attempts_remaining": max(0, 3 - attempts_used),
        "answered": _completed_count(rec),
        "total": len(pipeline._web_questions),
    }


def _start_mode_c_batch(rec: dict) -> dict:
    pipeline = rec["pipeline"]
    solve_map = _solve_map(pipeline)
    rec["diagnose_results"] = []
    queue: list[str] = []

    for question in pipeline._web_questions:
        first_attempt = (pipeline._web_user_answers.get(question.id) or "").strip()
        if not first_attempt:
            continue
        solve_result = solve_map.get(question.id)
        if not solve_result:
            continue
        correct_answer = solve_result.correct_answer.strip()
        diagnoser = ErrorDiagnoser(pipeline.llm, pipeline.logger, subject=pipeline.subject)
        is_correct = diagnoser._check_answer_correct(first_attempt, correct_answer, question.problem_type)
        work_info = pipeline._web_student_work_map.get(question.id) or {}
        if is_correct:
            rec["diagnose_results"].append(
                _make_mode_c_correct_result(question, solve_result, first_attempt, work_info=work_info)
            )
        else:
            queue.append(question.id)

    rec["mode_c_batch"] = {"queue": queue, "index": 0}
    if not queue:
        return _finalize_web_session(rec)
    return _start_mode_c_question(rec, queue[0])


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
        item = _serialize_question(q)
        item["supports_handwritten_work"] = pipeline.subject == "math"
        item["has_handwritten_work"] = q.id in (pipeline._web_student_work_map or {})
        out.append(item)
    return {"questions": out}


@app.post("/api/sessions/{session_id}/student-work")
async def api_sessions_student_work(
    session_id: str,
    question_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload and transcribe optional student handwritten work for a single question."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    rec = sessions[session_id]
    if rec["step"] not in ("solve_done", "answers_done", "diagnose_done"):
        raise HTTPException(status_code=400, detail="Student work can only be uploaded after solve step")

    pipeline = rec["pipeline"]
    question = next((q for q in pipeline._web_questions if q.id == question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
    if pipeline.subject != "math":
        raise HTTPException(status_code=400, detail="Handwritten work upload is only supported for math questions")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    suffix = Path(file.filename).suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use png, jpg, jpeg, or webp")

    student_work_dir = Path(pipeline.session_dir) / "student_work"
    student_work_dir.mkdir(parents=True, exist_ok=True)
    safe_question_id = question_id.replace("/", "_")
    saved_path = student_work_dir / f"{safe_question_id}_{uuid.uuid4().hex}{suffix}"
    content = await file.read()
    with open(saved_path, "wb") as f:
        f.write(content)

    work_info = _transcribe_handwritten_work_image(
        llm_client=pipeline.llm,
        image_path=str(saved_path),
        question_id=question_id,
    )
    pipeline._web_student_work_map[question_id] = work_info

    if work_info.get("error"):
        raise HTTPException(status_code=500, detail=work_info["error"])

    return {
        "ok": True,
        "question_id": question_id,
        "image_path": str(saved_path),
        "transcribed_work": work_info.get("transcribed_work", ""),
        "confidence": work_info.get("confidence", 0.0),
    }


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
        pipeline.set_diagnose_options_step(
            mode=pipeline._web_diagnose_mode,
            feedback_timing="per_question",
        )
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
    if any(r.question_id == question_id for r in rec.get("diagnose_results", [])):
        raise HTTPException(status_code=400, detail=f"Question {question_id} is already completed")
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
    answer = answer.strip()
    pipeline._web_user_answers[question_id] = answer
    diagnoser = ErrorDiagnoser(pipeline.llm, pipeline.logger, subject=pipeline.subject)
    diagnose_mode = pipeline._web_diagnose_mode
    student_work_text = (pipeline._web_student_work_map.get(question_id) or {}).get("transcribed_work") or None
    if diagnose_mode == "A":
        result, error = diagnoser.diagnose_mode_a(question, solve_result, answer)
    elif diagnose_mode == "C":
        correct_answer = solve_result.correct_answer.strip()
        is_correct = diagnoser._check_answer_correct(answer, correct_answer, question.problem_type)
        if is_correct:
            work_info = pipeline._web_student_work_map.get(question_id) or {}
            result = _make_mode_c_correct_result(question, solve_result, answer, work_info=work_info)
            error = None
            rec.setdefault("diagnose_results", []).append(result)
        else:
            payload = _start_mode_c_question(rec, question_id)
            answered = _completed_count(rec)
            return {
                **payload,
                "all_done": False,
                "answered": answered,
                "total": len(questions),
            }
    else:
        result, error = diagnoser.diagnose(
            question,
            solve_result,
            answer,
            student_work_text=student_work_text,
        )
        if result:
            rec.setdefault("diagnose_results", []).append(result)
    if error and not result:
        raise HTTPException(status_code=500, detail=error)
    if result:
        if diagnose_mode != "C" and student_work_text:
            work_info = pipeline._web_student_work_map.get(question_id) or {}
            result.student_work_image_path = work_info.get("image_path")
            result.student_work_transcription = work_info.get("transcribed_work")
    answered = _completed_count(rec)
    total = len(questions)
    all_done = answered >= total
    if all_done:
        _finalize_web_session(rec)
    return {
        "diagnose_result": result.model_dump() if result else None,
        "all_done": all_done,
        "answered": answered,
        "total": total,
    }


@app.post("/api/sessions/{session_id}/mode-c-second-attempt")
async def api_sessions_mode_c_second_attempt(
    session_id: str,
    question_id: str = Form(...),
    second_attempt: str = Form(...),
):
    """Submit the second attempt for a pending Mode C question."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    rec = sessions[session_id]
    pending = rec.get("mode_c_pending")
    if not pending:
        raise HTTPException(status_code=400, detail="No pending Mode C question")
    if pending.get("question_id") != question_id:
        raise HTTPException(status_code=400, detail="Question does not match pending Mode C step")

    pipeline = rec["pipeline"]
    question = next((q for q in pipeline._web_questions if q.id == question_id), None)
    solve_result = _solve_map(pipeline).get(question_id)
    if not question or not solve_result:
        raise HTTPException(status_code=404, detail="Question or solve result not found")

    work_info = pipeline._web_student_work_map.get(question_id) or {}
    student_work_text = work_info.get("transcribed_work") or None
    diagnoser = ErrorDiagnoser(pipeline.llm, pipeline.logger, subject=pipeline.subject)
    second_attempt = second_attempt.strip()
    attempts_used = int(pending.get("attempts_used", 1)) + 1
    correct_answer = solve_result.correct_answer.strip()
    is_correct = diagnoser._check_answer_correct(second_attempt, correct_answer, question.problem_type)

    if not is_correct and attempts_used < 3:
        next_payload = _start_mode_c_question(
            rec,
            question_id,
            first_attempt=pending.get("first_attempt", ""),
            current_attempt=second_attempt,
            attempts_used=attempts_used,
        )
        return {
            **next_payload,
            "all_done": False,
            "answered": _completed_count(rec),
            "total": len(pipeline._web_questions),
        }

    result, error = diagnoser.diagnose_after_second_attempt(
        question=question,
        solve_result=solve_result,
        first_attempt=pending.get("first_attempt", ""),
        second_attempt=second_attempt,
        student_work_text=student_work_text,
    )
    pipeline._web_user_answers[question_id] = second_attempt
    if error and not result:
        raise HTTPException(status_code=500, detail=error)
    if result and work_info:
        result.student_work_image_path = work_info.get("image_path")
        result.student_work_transcription = work_info.get("transcribed_work")
    if result:
        rec.setdefault("diagnose_results", []).append(result)
    rec["mode_c_pending"] = None

    answered = _completed_count(rec)
    total = len(pipeline._web_questions)

    batch_state = rec.get("mode_c_batch")
    if batch_state:
        batch_state["index"] += 1
        queue = batch_state.get("queue", [])
        if batch_state["index"] < len(queue):
            next_payload = _start_mode_c_question(rec, queue[batch_state["index"]])
            return {
                "diagnose_result": result.model_dump() if result else None,
                "all_done": False,
                "answered": answered,
                "total": total,
                "next_mode_c": next_payload,
            }
        rec["mode_c_batch"] = None

    all_done = answered >= total
    summary = _finalize_web_session(rec) if all_done else None
    return {
        "diagnose_result": result.model_dump() if result else None,
        "all_done": all_done,
        "answered": answered,
        "total": total,
        "summary": summary,
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
    if rec.get("mode_c_pending"):
        return rec["mode_c_pending"]
    if rec["step"] != "answers_done":
        raise HTTPException(status_code=400, detail="Expected step: answers_done")
    if rec["pipeline"]._web_diagnose_mode == "C":
        return _start_mode_c_batch(rec)
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
    return HTMLResponse(
        html.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store"
        },
    )


if __name__ == "__main__":
    import uvicorn
    # 127.0.0.1 可在本机浏览器直接打开；若需局域网访问再改为 0.0.0.0
    uvicorn.run(app, host="127.0.0.1", port=8000)
