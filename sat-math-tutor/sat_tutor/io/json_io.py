"""
JSON Input/Output Module
Handles reading and writing of structured data
"""

import json
import os
from typing import Any, Optional
from datetime import datetime

from ..core.models import (
    Question,
    SolveResult,
    DiagnoseResult,
    TranscribeOutput,
    SessionResult
)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Saves data to a JSON file
    
    Args:
        data: Data to be saved (supports Pydantic models)
        file_path: Output file path
        indent: Number of spaces for indentation
    """
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    
    # Handle Pydantic models
    if hasattr(data, 'model_dump'):
        data = data.model_dump()
    elif isinstance(data, list):
        data = [item.model_dump() if hasattr(item, 'model_dump') else item for item in data]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)


def load_json(file_path: str) -> Any:
    """
    Loads data from a JSON file
    
    Args:
        file_path: File path
    
    Returns:
        Parsed data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_transcribed(
    questions: list[Question], 
    file_path: str,
    pdf_name: Optional[str] = None,
    failed_pages: Optional[list[int]] = None,
    errors: Optional[list[str]] = None
) -> None:
# 必需字段
# 字段	类型	说明
# questions	array	题目列表
# questions[].id	string	题目ID，如 p1_q1
# questions[].source.pdf	string	PDF文件名
# questions[].source.page	int	页码
# questions[].problem_type	string	multiple_choice 或 numeric_entry
# questions[].stem	string	题干
# questions[].choices	object	选项 {"A": "...", "B": "...", ...}
    """Saves question extraction results"""
    data = {
        "questions": [q.model_dump() for q in questions],
        "total": len(questions),
        "timestamp": datetime.now().isoformat()
    }
    if pdf_name:
        data["pdf_name"] = pdf_name
    if failed_pages:
        data["failed_pages"] = failed_pages
    if errors:
        data["errors"] = errors
    save_json(data, file_path)


def load_transcribed(file_path: str) -> list[Question]:
    """Loads question extraction results"""
    data = load_json(file_path)
    questions_data = data.get("questions", data) if isinstance(data, dict) else data
    return [Question.model_validate(q) for q in questions_data]


def save_solve_results(results: list[SolveResult], file_path: str) -> None:
    """Saves solving results"""
    data = {
        "solve_results": [r.model_dump() for r in results],
        "total": len(results),
        "timestamp": datetime.now().isoformat()
    }
    save_json(data, file_path)


def load_solve_results(file_path: str) -> list[SolveResult]:
    """Loads solving results"""
    data = load_json(file_path)
    results_data = data.get("solve_results", data) if isinstance(data, dict) else data
    return [SolveResult.model_validate(r) for r in results_data]


def save_session_result(result: SessionResult, file_path: str) -> None:
    """Saves the complete session result"""
    save_json(result.model_dump(), file_path)


def load_session_result(file_path: str) -> SessionResult:
    """Loads session result"""
    data = load_json(file_path)
    return SessionResult.model_validate(data)


def create_session_output(
    session_id: str,
    pdf_path: str,
    mode: str,
    questions: list[Question],
    failed_pages: list[int],
    errors: list[str],
    solve_results: Optional[list[SolveResult]] = None,
    diagnose_results: Optional[list[DiagnoseResult]] = None,
    user_answers: Optional[dict[str, str]] = None
) -> SessionResult:
    """
    Creates a session output object
    
    Args:
        session_id: Session ID
        pdf_path: PDF file path
        mode: Running mode
        questions: Extracted questions
        failed_pages: List of failed page numbers
        errors: Error messages
        solve_results: Solving results
        diagnose_results: Diagnosis results
        user_answers: User answers
    
    Returns:
        SessionResult object
    """
    # Calculate statistics
    total_questions = len(questions)
    answered_questions = len(user_answers) if user_answers else 0
    correct_count = 0
    incorrect_ids = []
    first_attempt_wrong_count = 0
    first_attempt_wrong_ids = []
    
    if diagnose_results:
        for dr in diagnose_results:
            if dr.is_correct:
                correct_count += 1
            else:
                incorrect_ids.append(dr.question_id)
            
            # Track first attempt wrong (Mode C scaffolded tutoring)
            if dr.first_attempt_wrong:
                first_attempt_wrong_count += 1
                first_attempt_wrong_ids.append(dr.question_id)
    
    return SessionResult(
        session_id=session_id,
        pdf_path=pdf_path,
        mode=mode,
        timestamp=datetime.now().isoformat(),
        transcribed=TranscribeOutput(
            questions=questions,
            total_pages=len(set(q.source.page for q in questions)) if questions else 0,
            failed_pages=failed_pages,
            errors=errors
        ),
        solve_results=solve_results or [],
        diagnose_results=diagnose_results or [],
        total_questions=total_questions,
        answered_questions=answered_questions,
        correct_count=correct_count,
        incorrect_ids=incorrect_ids,
        first_attempt_wrong_count=first_attempt_wrong_count,
        first_attempt_wrong_ids=first_attempt_wrong_ids
    )