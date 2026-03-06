"""
Main Processing Pipeline
Coordinates execution of each stage
"""

import os
from typing import Optional, Literal
from pathlib import Path

from ..llm.base import LLMClient
from ..llm.openai_client import OpenAIClient
from ..llm.mock_client import MockLLMClient
from ..ingest.pdf_to_images import pdf_to_images, PDFConversionError
from ..ingest.vision_extract import VisionQuestionExtractor
from ..ingest.ocr_extract import OCRExtractor, OCR_AVAILABLE
from ..ingest.text_extract import TextQuestionExtractor
from .solver import QuestionSolver
from .diagnose import ErrorDiagnoser
from .models import SessionResult, SolveResult, Question, DiagnoseResult
from ..io.json_io import (
    save_json,
    save_transcribed,
    save_session_result,
    create_session_output
)
from ..io.report_md import save_report_md, print_summary
from ..io.answers import (
    collect_answers_interactive, 
    load_answers_from_json,
    ask_correct_answers_choice,
    ask_user_answers_choice,
    ask_diagnose_mode,
    collect_second_attempt
)
from ..io.student_simulator import ask_simulate_student
from ..utils.logging import Logger, create_session_logger
from ..utils.time import generate_session_id


RunMode = Literal["transcribe_only", "solve", "diagnose"]
SubjectType = Literal["math", "english"]


def load_correct_answers_as_solve_results(
    json_path: str, 
    questions: list[Question]
) -> list[SolveResult]:
    """
    Load answers from correct answers JSON file, convert to SolveResult list
    
    Args:
        json_path: Correct answers JSON file path
        questions: Question list (for question info)
    
    Returns:
        SolveResult list
    """
    import json
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    question_map = {q.id: q for q in questions}
    
    solve_results = []
    
    # Support two formats:
    # 1. Simple: {"p1_q1": "A", "p1_q2": "B"}
    # 2. Detailed: {"p1_q1": {"answer": "A", "topic": "algebra", "steps": [...]}}
    
    for q_id, value in data.items():
        if q_id.startswith("_"):
            continue
        if q_id not in question_map:
            continue
        
        question = question_map[q_id]
        
        if isinstance(value, dict):
            solve_results.append(SolveResult(
                question_id=q_id,
                correct_answer=str(value.get("answer", value.get("correct_answer", ""))),
                topic=value.get("topic", "unknown"),
                key_steps=value.get("steps", value.get("key_steps", ["Standard answer (no solution steps)"])),
                final_reason=value.get("reason", value.get("final_reason", "From correct answers file")),
                confidence=value.get("confidence", 1.0)
            ))
        else:
            solve_results.append(SolveResult(
                question_id=q_id,
                correct_answer=str(value),
                topic="unknown",
                key_steps=["Standard answer (no solution steps)"],
                final_reason="From correct answers file",
                confidence=1.0
            ))
    
    return solve_results


class GREMathPipeline:
    """
    SAT Problem Processing Pipeline
    Coordinates PDF processing, transcription, solving, diagnosis stages
    Supports both Math (vision-based) and English (OCR-based) extraction
    """
    
    def __init__(
        self,
        use_mock: bool = False,
        output_dir: str = "outputs",
        subject: SubjectType = "math"
    ):
        """
        Initialize pipeline
        
        Args:
            use_mock: Whether to use mock client
            output_dir: Output directory
            subject: Subject type - "math" (vision) or "english" (OCR)
        """
        self.output_dir = output_dir
        self.use_mock = use_mock
        self.subject = subject
        
        if use_mock:
            self.llm: LLMClient = MockLLMClient()
        else:
            openai_client = OpenAIClient()
            if openai_client.is_available:
                self.llm = openai_client
            else:
                print("Warning: OpenAI API Key not configured, using Mock mode")
                self.llm = MockLLMClient()
        
        self.session_id: Optional[str] = None
        self.session_dir: Optional[str] = None
        self.logger: Optional[Logger] = None
    
    def _init_session(self) -> None:
        """Initialize session"""
        self.session_id = generate_session_id()
        self.session_dir = os.path.join(self.output_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "pages"), exist_ok=True)
        self.logger = create_session_logger(self.session_dir)
    
    def _load_transcribed(self, json_path: str) -> tuple[list[Question], list[int], list[str], str]:
        """
        Load questions from existing transcribed.json file
        
        Args:
            json_path: Path to transcribed.json file
            
        Returns:
            Tuple of (questions, failed_pages, errors, pdf_name)
        """
        import json
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse questions
        questions = []
        for q_data in data.get("questions", []):
            try:
                questions.append(Question(**q_data))
            except Exception as e:
                self.logger.warning(f"Failed to parse question: {e}")
        
        failed_pages = data.get("failed_pages", [])
        errors = data.get("errors", [])
        pdf_name = data.get("pdf_name", "unknown.pdf")
        
        return questions, failed_pages, errors, pdf_name
    
    def run(
        self,
        pdf_path: str,
        mode: RunMode = "diagnose",
        pages: str = "all",
        dpi: int = 300,
        answers_json: Optional[str] = None,
        correct_answers_json: Optional[str] = None,
        interactive: bool = True,
        transcribed_json: Optional[str] = None
    ) -> SessionResult:
        """
        Run complete pipeline
        
        Args:
            pdf_path: PDF file path
            mode: Run mode
            pages: Page range
            dpi: Image resolution
            answers_json: User answers JSON file path (CLI preset)
            correct_answers_json: Correct answers JSON file path (CLI preset)
            interactive: Enable interactive selection (default True)
            transcribed_json: Load existing transcribed.json file (skip extraction)
        
        Returns:
            SessionResult object
        """
        self._init_session()
        
        self.logger.info(f"Run mode: {mode}")
        self.logger.info(f"Subject: {self.subject}")
        
        # Check if loading from existing transcribed file
        if transcribed_json:
            self.logger.info(f"Loading transcribed file: {transcribed_json}")
            questions, failed_pages, errors, pdf_name = self._load_transcribed(transcribed_json)
            # Use pdf_path if provided, otherwise use pdf_name from transcribed file
            if not pdf_path:
                pdf_path = pdf_name
            self.logger.info(f"Loaded {len(questions)} questions from transcribed file")
        else:
            self.logger.info(f"Starting to process PDF: {pdf_path}")
            self.logger.info(f"Page range: {pages}")
            
            pdf_name = os.path.basename(pdf_path)
            
            # ===== Stage 0: PDF to Images =====
            self.logger.info("Stage 0: PDF to Images")
            
            try:
                pages_dir = os.path.join(self.session_dir, "pages")
                image_paths = pdf_to_images(
                    pdf_path=pdf_path,
                    output_dir=pages_dir,
                    pages=pages,
                    dpi=dpi
                )
                self.logger.info(f"Successfully converted {len(image_paths)} pages")
            except PDFConversionError as e:
                self.logger.error(f"PDF conversion failed: {e}")
                raise
            
            # ===== Stage T: Transcribe =====
            self.logger.info("Stage T: Transcribe")
            
            if self.subject == "english":
                # English: OCR + Text LLM extraction
                questions, failed_pages, errors = self._extract_english_questions(
                    image_paths=image_paths,
                    pdf_name=pdf_name
                )
            else:
                # Math: Vision LLM extraction
                extractor = VisionQuestionExtractor(self.llm, self.logger)
                questions, failed_pages, errors = extractor.extract_from_images(
                    image_paths=image_paths,
                    pdf_name=pdf_name
                )
            
            self.logger.info(f"Successfully extracted {len(questions)} questions")
        if failed_pages:
            self.logger.warning(f"Failed pages: {failed_pages}")
        
        transcribed_path = os.path.join(self.session_dir, "transcribed.json")
        save_transcribed(
            questions, 
            transcribed_path,
            pdf_name=os.path.basename(pdf_path) if pdf_path else None,
            failed_pages=failed_pages,
            errors=errors
        )
        
        if mode == "transcribe_only":
            result = create_session_output(
                session_id=self.session_id,
                pdf_path=pdf_path,
                mode=mode,
                questions=questions,
                failed_pages=failed_pages,
                errors=errors
            )
            self._save_and_print(result)
            return result
        
        # ===== Stage S: Solve =====
        use_correct_file = correct_answers_json
        
        if interactive and not correct_answers_json and mode == "diagnose":
            use_correct_file = ask_correct_answers_choice(questions)
        
        if use_correct_file:
            self.logger.info("Stage S: Using correct answers file")
            try:
                solve_results = load_correct_answers_as_solve_results(
                    use_correct_file, questions
                )
                self.logger.info(f"Loaded {len(solve_results)} answers from correct answers file")
                
                loaded_ids = {sr.question_id for sr in solve_results}
                missing_questions = [q for q in questions if q.id not in loaded_ids]
                
                if missing_questions:
                    missing_ids = [q.id for q in missing_questions]
                    self.logger.warning(f"Following questions not found in correct answers file, will use LLM to solve: {missing_ids}")
                    
                    solver = QuestionSolver(self.llm, self.logger)
                    extra_results, extra_errors = solver.solve_batch(missing_questions)
                    
                    solve_results.extend(extra_results)
                    errors.extend(extra_errors)
                    
                    self.logger.info(f"LLM supplemented {len(extra_results)} questions")
                    
            except Exception as e:
                self.logger.error(f"Failed to load correct answers file: {e}")
                raise ValueError(f"Cannot load correct answers file: {e}")
        else:
            self.logger.info("Stage S: LLM Solving")
            
            solver = QuestionSolver(self.llm, self.logger)
            solve_results, solve_errors = solver.solve_batch(questions)
            
            self.logger.info(f"Successfully solved {len(solve_results)} questions")
            errors.extend(solve_errors)
        
        if mode == "solve":
            result = create_session_output(
                session_id=self.session_id,
                pdf_path=pdf_path,
                mode=mode,
                questions=questions,
                failed_pages=failed_pages,
                errors=errors,
                solve_results=solve_results
            )
            self._save_and_print(result)
            return result
        
        # ===== Collect User Answers =====
        self.logger.info("Collecting user answers")
        
        student_work_map: dict[str, dict] = {}
        
        if answers_json:
            try:
                user_answers = load_answers_from_json(answers_json)
                self.logger.info(f"Loaded {len(user_answers)} user answers from {answers_json}")
            except Exception as e:
                self.logger.error(f"Failed to load user answers file: {e}")
                user_answers = {}
        elif interactive:
            user_answers, student_work_map = ask_user_answers_choice(
                questions, 
                llm_client=self.llm,
                solve_results=solve_results,
                session_dir=self.session_dir,
                subject=self.subject
            )
        else:
            user_answers = {}
        
        if student_work_map:
            work_path = os.path.join(self.session_dir, "student_work_transcriptions.json")
            save_json(student_work_map, work_path)
            self.logger.info(f"Saved handwritten work transcriptions: {work_path}")
        
        # ===== Select Diagnosis Mode =====
        diagnose_mode = "B"  # Default: Contrastive
        if interactive:
            diagnose_mode = ask_diagnose_mode()
            self.logger.info(f"Selected diagnosis mode: {diagnose_mode}")
        
        # ===== Stage D: Diagnose =====
        self.logger.info(f"Stage D: Diagnose (Mode {diagnose_mode})")
        
        diagnoser = ErrorDiagnoser(self.llm, self.logger, subject=self.subject)
        
        # Mode C requires special handling (scaffolded tutoring)
        if diagnose_mode == "C" and interactive:
            diagnose_results, diagnose_errors = self._diagnose_mode_c(
                diagnoser=diagnoser,
                questions=questions,
                solve_results=solve_results,
                user_answers=user_answers,
                student_work_map=student_work_map
            )
        else:
            diagnose_results, diagnose_errors = diagnoser.diagnose_batch(
                questions=questions,
                solve_results=solve_results,
                user_answers=user_answers,
                mode=diagnose_mode,
                student_work_map=student_work_map
            )
        
        self.logger.info(f"Completed diagnosis for {len(diagnose_results)} questions")
        errors.extend(diagnose_errors)
        
        result = create_session_output(
            session_id=self.session_id,
            pdf_path=pdf_path,
            mode=mode,
            questions=questions,
            failed_pages=failed_pages,
            errors=errors,
            solve_results=solve_results,
            diagnose_results=diagnose_results,
            user_answers=user_answers
        )
        
        self._save_and_print(result)
        return result
    
    def _diagnose_mode_c(
        self,
        diagnoser: ErrorDiagnoser,
        questions: list[Question],
        solve_results: list[SolveResult],
        user_answers: dict[str, str],
        student_work_map: Optional[dict[str, dict]] = None
    ) -> tuple[list[DiagnoseResult], list[str]]:
        """
        Mode C: Scaffolded Tutoring
        
        For each wrong answer:
        1. Show hints (without revealing answer)
        2. Let student try again
        3. Give full explanation
        
        Args:
            diagnoser: ErrorDiagnoser instance
            questions: List of Question objects
            solve_results: List of SolveResult objects
            user_answers: Dictionary of user answers
        
        Returns:
            (List of DiagnoseResult, List of error messages)
        """
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console(width=100)
        results = []
        errors = []
        
        # Build a mapping of solving results
        solve_map = {sr.question_id: sr for sr in solve_results}
        
        for question in questions:
            if question.id not in user_answers:
                continue
            
            first_answer = user_answers[question.id]
            if not first_answer:
                continue
            
            work_info = (student_work_map or {}).get(question.id, {})
            student_work_text = work_info.get("transcribed_work") or None
            
            solve_result = solve_map.get(question.id)
            if not solve_result:
                errors.append(f"Missing solving result for question {question.id}")
                continue
            
            correct_answer = solve_result.correct_answer.strip()
            is_correct = diagnoser._check_answer_correct(
                first_answer, correct_answer, question.problem_type
            )
            
            # If correct on first try, just record success
            if is_correct:
                console.print(f"\n[green]Question {question.id}: Correct on first try![/green]")
                results.append(DiagnoseResult(
                    question_id=question.id,
                    user_answer=first_answer,
                    correct_answer=correct_answer,
                    is_correct=True,
                    student_work_image_path=work_info.get("image_path"),
                    student_work_transcription=work_info.get("transcribed_work"),
                    why_user_choice_is_tempting=None,
                    likely_misconceptions=[],
                    how_to_get_correct=None,
                    option_analysis=[]
                ))
                continue
            
            # Wrong answer - start scaffolded tutoring
            console.print(f"\n[yellow]Question {question.id}: Not quite right. Let's work through this...[/yellow]")

            
            # Step 1: Get hints
            self.logger.info(f"[Mode C] Getting hints for {question.id}")
            hint_result = diagnoser.get_hint_for_wrong_answer(
                question=question,
                solve_result=solve_result,
                user_answer=first_answer,
                student_work_text=student_work_text
            )
            
            # Step 2: Keep retrying until correct.
            # Reuse the original actionable hints, but refresh error analysis
            # for each new wrong answer.
            current_answer = first_answer
            final_answer = first_answer
            while True:
                next_answer = collect_second_attempt(
                    question=question,
                    first_answer=current_answer,
                    hint_result=hint_result
                )
                
                final_answer = next_answer
                is_correct_now = diagnoser._check_answer_correct(
                    final_answer, correct_answer, question.problem_type
                )
                
                if is_correct_now:
                    break
                
                # Refresh only error analysis based on the latest wrong answer.
                # Keep actionable hints unchanged (as requested).
                refreshed_hint = diagnoser.get_hint_for_wrong_answer(
                    question=question,
                    solve_result=solve_result,
                    user_answer=final_answer,
                    student_work_text=student_work_text
                )
                if refreshed_hint:
                    hint_result["error_analysis"] = refreshed_hint.get(
                        "error_analysis",
                        hint_result.get("error_analysis", "")
                    )
                
                current_answer = final_answer
                console.print("\n[yellow]Still not correct. Review the same hints and try again.[/yellow]")
            
            # Step 3: Final diagnosis (called once, after student gets correct)
            self.logger.info(f"[Mode C] Final diagnosis for {question.id}")
            result, error = diagnoser.diagnose_after_second_attempt(
                question=question,
                solve_result=solve_result,
                first_attempt=first_answer,
                second_attempt=final_answer,
                student_work_text=student_work_text
            )
            
            if result:
                if work_info:
                    result.student_work_image_path = work_info.get("image_path")
                    result.student_work_transcription = work_info.get("transcribed_work")
                results.append(result)
                
                # Show final result
                if result.is_correct:
                    console.print(f"\n[green]Excellent! You got it right on the second try![/green]")
                else:
                    console.print(f"\n[red]The correct answer is: {correct_answer}[/red]")
                
                # Show explanation
                if result.how_to_get_correct:
                    console.print(Panel(
                        result.how_to_get_correct,
                        title="[bold]Complete Explanation[/bold]",
                        border_style="cyan"
                    ))
            
            if error:
                errors.append(error)
        
        return results, errors
    
    def _extract_english_questions(
        self,
        image_paths: list[str],
        pdf_name: str
    ) -> tuple[list[Question], list[int], list[str]]:
        """
        Extract English questions using OCR + Text LLM
        
        Args:
            image_paths: List of image file paths
            pdf_name: PDF file name
        
        Returns:
            (questions, failed_pages, errors)
        """
        if not OCR_AVAILABLE:
            error_msg = (
                "OCR is not available. Install with:\n"
                "  pip install pytesseract Pillow\n"
                "  brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)"
            )
            self.logger.error(error_msg)
            raise ImportError(error_msg)
        
        # Step 1: OCR - Extract text from images
        self.logger.info("Step 1: OCR text extraction")
        ocr_extractor = OCRExtractor(lang="eng", logger=self.logger)
        page_texts, ocr_failed, ocr_errors = ocr_extractor.extract_text_from_images(image_paths)
        
        if ocr_failed:
            self.logger.warning(f"OCR failed for pages: {ocr_failed}")
        
        # Save OCR text for debugging
        ocr_text_path = os.path.join(self.session_dir, "ocr_text.txt")
        combined_text = ocr_extractor.combine_texts(page_texts)
        with open(ocr_text_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        self.logger.info(f"OCR text saved to: {ocr_text_path}")
        
        # Step 2: LLM - Extract questions from OCR text
        self.logger.info("Step 2: LLM question extraction from OCR text")
        text_extractor = TextQuestionExtractor(self.llm, self.logger)
        questions, extract_failed, extract_errors = text_extractor.extract_from_page_texts(
            page_texts=page_texts,
            pdf_name=pdf_name
        )
        
        # Combine failures and errors
        failed_pages = list(set(ocr_failed + extract_failed))
        errors = ocr_errors + extract_errors
        
        return questions, failed_pages, errors
    
    def _save_and_print(self, result: SessionResult) -> None:
        """Save results and print summary"""
        results_path = os.path.join(self.session_dir, "results.json")
        save_session_result(result, results_path)
        
        report_path = os.path.join(self.session_dir, "report.md")
        save_report_md(result, report_path)
        
        print_summary(result)
        
        self.logger.info(f"Results saved to: {self.session_dir}")
        self.logger.close()
