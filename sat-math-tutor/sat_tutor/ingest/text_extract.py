"""
Text-based Question Extraction Module
Uses LLM to extract questions from OCR text
"""

import os
import re
from typing import Optional

from ..llm.base import LLMClient
from ..llm.prompts import (
    ENGLISH_TRANSCRIBE_SYSTEM_PROMPT,
    ENGLISH_TRANSCRIBE_USER_PROMPT_TEMPLATE,
    ENGLISH_QUESTION_SCHEMA_HINT
)
from ..core.models import Question
from ..core.validators import validate_questions_list
from ..utils.logging import Logger


def normalize_question_id(q_id: str, page_number: int, question_index: int) -> str:
    """
    Normalize question ID
    
    Args:
        q_id: Original ID
        page_number: Page number
        question_index: Question index on page (1-based)
    
    Returns:
        Normalized ID in format p{page}_q{num}
    """
    if '.' in q_id:
        match = re.match(r'p(\d+)_q(\d+)(?:\.(\d+))?', q_id)
        if match:
            return f"p{page_number}_q{question_index}"
    
    if re.match(r'^p\d+_q\d+$', q_id):
        return q_id
    
    return f"p{page_number}_q{question_index}"


class TextQuestionExtractor:
    """
    Text-based Question Extractor
    Extracts structured questions from OCR text using LLM
    """
    
    def __init__(self, llm_client: LLMClient, logger: Optional[Logger] = None):
        """
        Initialize extractor
        
        Args:
            llm_client: LLM client
            logger: Logger instance
        """
        self.llm = llm_client
        self.logger = logger
    
    def _log(self, message: str, level: str = "info"):
        """Log a message"""
        if self.logger:
            self.logger.log(message, level)
    
    def extract_from_text(
        self,
        text: str,
        pdf_name: str,
        page_number: int,
        retry_on_failure: bool = True
    ) -> tuple[list[Question], Optional[str]]:
        """
        Extract questions from OCR text of a single page
        
        Args:
            text: OCR extracted text
            pdf_name: PDF file name
            page_number: Page number
            retry_on_failure: Whether to retry on parse failure
        
        Returns:
            (List of Questions, Error message or None)
        """
        if not text.strip():
            return [], f"Empty text for page {page_number}"
        
        self._log(f"Extracting questions from page {page_number} text...")
        
        # Build user prompt
        user_prompt = ENGLISH_TRANSCRIBE_USER_PROMPT_TEMPLATE.format(
            pdf_name=pdf_name,
            page_number=page_number,
            ocr_text=text
        )
        
        # First attempt
        response = self.llm.generate_json(
            system_prompt=ENGLISH_TRANSCRIBE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_hint=ENGLISH_QUESTION_SCHEMA_HINT,
            temperature=0.1
        )
        
        if not response.success:
            self._log(f"LLM call failed: {response.error}", "error")
            return [], response.error
        
        # Try to parse response
        result = validate_questions_list(response.content)
        
        if result.success:
            questions = result.data
            for idx, q in enumerate(questions, 1):
                q.source.pdf = pdf_name
                q.source.page = page_number
                q.exam = "SAT"
                q.section = "English"
                old_id = q.id
                q.id = normalize_question_id(q.id, page_number, idx)
                if old_id != q.id:
                    self._log(f"Normalized question ID: {old_id} -> {q.id}", "warning")
            self._log(f"Successfully extracted {len(questions)} questions from page {page_number}")
            return questions, None
        
        # Parse failed, retry
        if retry_on_failure:
            self._log(f"First parse failed, retrying... Error: {result.error}", "warning")
            
            response = self.llm.generate_json(
                system_prompt=ENGLISH_TRANSCRIBE_SYSTEM_PROMPT,
                user_prompt=user_prompt + "\n\nIMPORTANT: Output ONLY valid JSON, no other text!",
                schema_hint=ENGLISH_QUESTION_SCHEMA_HINT,
                temperature=0.0
            )
            
            if response.success:
                result = validate_questions_list(response.content)
                if result.success:
                    questions = result.data
                    for idx, q in enumerate(questions, 1):
                        q.source.pdf = pdf_name
                        q.source.page = page_number
                        q.exam = "SAT"
                        q.section = "English"
                        old_id = q.id
                        q.id = normalize_question_id(q.id, page_number, idx)
                        if old_id != q.id:
                            self._log(f"Normalized question ID: {old_id} -> {q.id}", "warning")
                    self._log(f"Retry successful, extracted {len(questions)} questions from page {page_number}")
                    return questions, None
        
        error_msg = f"Failed to parse page {page_number}: {result.error}\nRaw response: {response.content[:500]}..."
        self._log(error_msg, "error")
        return [], error_msg
    
    def extract_from_page_texts(
        self,
        page_texts: dict[int, str],
        pdf_name: str
    ) -> tuple[list[Question], list[int], list[str]]:
        """
        Extract questions from multiple pages of OCR text
        
        Args:
            page_texts: Dict of page_number -> OCR text
            pdf_name: PDF file name
        
        Returns:
            (List of all questions, List of failed pages, List of errors)
        """
        all_questions = []
        failed_pages = []
        errors = []
        
        for page_num in sorted(page_texts.keys()):
            text = page_texts[page_num]
            
            questions, error = self.extract_from_text(
                text=text,
                pdf_name=pdf_name,
                page_number=page_num
            )
            
            if questions:
                all_questions.extend(questions)
            
            if error:
                failed_pages.append(page_num)
                errors.append(error)
        
        return all_questions, failed_pages, errors

