"""
Solver Module (Stage S)
Solve extracted questions
"""

from typing import Optional

from ..llm.base import LLMClient
from ..llm.prompts import (
    SOLVE_SYSTEM_PROMPT,
    SOLVE_USER_PROMPT_TEMPLATE,
    SOLVE_SCHEMA_HINT
)
from .models import Question, SolveResult
from .validators import validate_solve_result, extract_json_from_text
from ..utils.logging import Logger

import json


class QuestionSolver:
    """
    Question Solver
    Use LLM to solve SAT math questions
    """
    
    def __init__(self, llm_client: LLMClient, logger: Optional[Logger] = None):
        """
        Initialize solver
        
        Args:
            llm_client: LLM client
            logger: Logger
        """
        self.llm = llm_client
        self.logger = logger
    
    def _log(self, message: str, level: str = "info"):
        """Log message"""
        if self.logger:
            self.logger.log(message, level)
    
    def solve(self, question: Question, retry_on_failure: bool = True) -> tuple[Optional[SolveResult], Optional[str]]:
        """
        Solve single question
        
        Args:
            question: Question object
            retry_on_failure: Whether to retry on failure
        
        Returns:
            (solve result, error message or None)
        """
        self._log(f"Solving question {question.id}...")
        
        # Prepare options
        choices = question.choices
        choice_a = choices.get("A", "N/A")
        choice_b = choices.get("B", "N/A")
        choice_c = choices.get("C", "N/A")
        choice_d = choices.get("D", "N/A")
        choice_e = choices.get("E", "N/A")
        
        # Prepare LaTeX info
        latex_info = ""
        if question.latex_equations:
            latex_info = f"Related formulas: {', '.join(question.latex_equations)}"
        
        # Prepare diagram description
        diagram_info = ""
        if question.diagram_description:
            diagram_info = f"Diagram description: {question.diagram_description}"
        
        # Build user prompt
        user_prompt = SOLVE_USER_PROMPT_TEMPLATE.format(
            question_id=question.id,
            problem_type=question.problem_type,
            stem=question.stem,
            choice_a=choice_a,
            choice_b=choice_b,
            choice_c=choice_c,
            choice_d=choice_d,
            choice_e=choice_e,
            latex_info=latex_info,
            diagram_info=diagram_info
        )
        
        # Call LLM (use low temperature to ensure answer accuracy)
        response = self.llm.generate_json(
            system_prompt=SOLVE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_hint=SOLVE_SCHEMA_HINT,
            temperature=0.0  # Use lowest temperature for deterministic answers
        )
        
        if not response.success:
            self._log(f"LLM call failed: {response.error}", "error")
            return None, response.error
        
        # Parse response
        result = validate_solve_result(response.content)
        
        if result.success:
            solve_result = result.data
            solve_result.question_id = question.id
            self._log(f"Question {question.id} solved, answer: {solve_result.correct_answer}")
            return solve_result, None
        
        # Parse failed, try retry
        if retry_on_failure:
            self._log(f"First parse failed, retrying... Error: {result.error}", "warning")
            
            response = self.llm.generate_json(
                system_prompt=SOLVE_SYSTEM_PROMPT,
                user_prompt=user_prompt + "\n\nPlease output strict JSON format, no other text.",
                schema_hint=SOLVE_SCHEMA_HINT,
                temperature=0.0
            )
            
            if response.success:
                result = validate_solve_result(response.content)
                if result.success:
                    solve_result = result.data
                    solve_result.question_id = question.id
                    self._log(f"Retry succeeded, question {question.id} answer: {solve_result.correct_answer}")
                    return solve_result, None
        
        # Final failure, try manual parse
        try:
            extracted = extract_json_from_text(response.content)
            if extracted:
                data = json.loads(extracted)
                solve_result = SolveResult(
                    question_id=question.id,
                    correct_answer=data.get("correct_answer", "C"),
                    topic=data.get("topic", "unknown"),
                    key_steps=data.get("key_steps", ["Parse failed"]),
                    final_reason=data.get("final_reason", "Parse failed"),
                    confidence=data.get("confidence", 0.5)
                )
                return solve_result, None
        except Exception:
            pass
        
        error_msg = f"Question {question.id} solve parse failed: {result.error}"
        self._log(error_msg, "error")
        return None, error_msg
    
    def solve_batch(self, questions: list[Question]) -> tuple[list[SolveResult], list[str]]:
        """
        Batch solve questions
        
        Args:
            questions: Question list
        
        Returns:
            (solve results list, errors list)
        """
        results = []
        errors = []
        
        for question in questions:
            result, error = self.solve(question)
            if result:
                results.append(result)
            if error:
                errors.append(error)
        
        return results, errors
