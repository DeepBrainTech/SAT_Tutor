"""
Visual Question Extraction Module (Stage T)
Uses vision models to extract questions from images
"""

import os
from typing import Optional
from pathlib import Path

from ..llm.base import LLMClient
from ..llm.prompts import (
    TRANSCRIBE_SYSTEM_PROMPT,
    TRANSCRIBE_USER_PROMPT_TEMPLATE,
    TRANSCRIBE_RETRY_PROMPT,
    QUESTION_SCHEMA_HINT
)
from ..core.models import Question
from ..core.validators import validate_questions_list, extract_json_from_text
from ..utils.logging import Logger
import re


def normalize_question_id(q_id: str, page_number: int, question_index: int) -> str:
    """
    Normalizes the question ID and fixes potential formatting issues (e.g., decimals).
    
    Args:
        q_id: Original ID
        page_number: Page number
        question_index: Sequence index on current page (1-based)
    
    Returns:
        Normalized ID in the format p{page}_q{num}
    """
    # Check if it contains a decimal point (e.g., p1_q1.1)
    if '.' in q_id:
        # Remove the decimal point and subsequent parts, or regenerate the ID
        # Attempt to extract page number and base question number
        match = re.match(r'p(\d+)_q(\d+)(?:\.(\d+))?', q_id)
        if match:
            page = match.group(1)
            base_num = match.group(2)
            sub_num = match.group(3)
            
            # If there are sub-numbers (e.g., 1.1, 1.2), generate a new sequential ID
            # Use the passed question_index to ensure uniqueness
            return f"p{page_number}_q{question_index}"
    
    # Validate if format is correct
    if re.match(r'^p\d+_q\d+$', q_id):
        return q_id
    
    # Format incorrect, regenerate
    return f"p{page_number}_q{question_index}"


class VisionQuestionExtractor:
    """
    Vision Question Extractor
    Extracts structured SAT math questions from images
    """
    
    def __init__(self, llm_client: LLMClient, logger: Optional[Logger] = None):
        """
        Initializes the extractor
        
        Args:
            llm_client: LLM client
            logger: Logger instance
        """
        self.llm = llm_client
        self.logger = logger
    
    def _log(self, message: str, level: str = "info"):
        """Logs a message"""
        if self.logger:
            self.logger.log(message, level)
    
    def extract_from_image(
        self,
        image_path: str,
        pdf_name: str,
        page_number: int,
        retry_on_failure: bool = True
    ) -> tuple[list[Question], Optional[str]]:
        """
        Extracts questions from a single image
        
        Args:
            image_path: Path to the image
            pdf_name: Name of the PDF file
            page_number: Page number
            retry_on_failure: Whether to retry on parsing failure
        
        Returns:
            (List of Questions, Error message or None)
        """
        if not os.path.exists(image_path):
            return [], f"Image file does not exist: {image_path}"
        
        self._log(f"Extracting questions from page {page_number}...")
        
        # Construct user prompt
        user_prompt = TRANSCRIBE_USER_PROMPT_TEMPLATE.format(
            pdf_name=pdf_name,
            page_number=page_number
        )
        
        # First attempt
        response = self.llm.generate_json(
            system_prompt=TRANSCRIBE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_hint=QUESTION_SCHEMA_HINT,
            images=[image_path],
            temperature=0.1
        )
        
        if not response.success:
            self._log(f"LLM call failed: {response.error}", "error")
            return [], response.error
        
        # Attempt to parse response
        result = validate_questions_list(response.content)
        
        if result.success:
            questions = result.data
            # Ensure source info is correct and normalize IDs
            for idx, q in enumerate(questions, 1):
                q.source.pdf = pdf_name
                q.source.page = page_number
                # Normalize ID, fix decimal points, etc.
                old_id = q.id
                q.id = normalize_question_id(q.id, page_number, idx)
                if old_id != q.id:
                    self._log(f"Normalized question ID: {old_id} → {q.id}", "warning")
            self._log(f"Successfully extracted {len(questions)} questions from page {page_number}")
            return questions, None
        
        # Parsing failed, attempt retry
        if retry_on_failure:
            self._log(f"First parsing attempt failed, retrying... Error: {result.error}", "warning")
            
            retry_prompt = TRANSCRIBE_RETRY_PROMPT.format(
                page=page_number,
                pdf_name=pdf_name
            )
            
            response = self.llm.generate_json(
                system_prompt=TRANSCRIBE_SYSTEM_PROMPT,
                user_prompt=retry_prompt,
                schema_hint=QUESTION_SCHEMA_HINT,
                images=[image_path],
                temperature=0.0  # Lower temperature for higher determinism
            )
            
            if response.success:
                result = validate_questions_list(response.content)
                if result.success:
                    questions = result.data
                    for idx, q in enumerate(questions, 1):
                        q.source.pdf = pdf_name
                        q.source.page = page_number
                        # Normalize ID
                        old_id = q.id
                        q.id = normalize_question_id(q.id, page_number, idx)
                        if old_id != q.id:
                            self._log(f"Normalized question ID: {old_id} → {q.id}", "warning")
                    self._log(f"Retry successful, extracted {len(questions)} questions from page {page_number}")
                    return questions, None
        
        # Final failure
        error_msg = f"Failed to parse page {page_number}: {result.error}\nOriginal response: {response.content[:500]}..."
        self._log(error_msg, "error")
        return [], error_msg
    
    def extract_from_images(
        self,
        image_paths: list[str],
        pdf_name: str,
        start_page: int = 1
    ) -> tuple[list[Question], list[int], list[str]]:
        """
        Extracts questions from multiple images
        
        Args:
            image_paths: List of image paths
            pdf_name: Name of the PDF file
            start_page: Starting page number
        
        Returns:
            (List of all questions, List of failed page numbers, List of error messages)
        """
        all_questions = []
        failed_pages = []
        errors = []
        
        for i, image_path in enumerate(image_paths):
            # Extract page number from filename
            page_num = start_page + i
            filename = Path(image_path).stem
            if filename.startswith("page_"):
                try:
                    page_num = int(filename.split("_")[1])
                except (IndexError, ValueError):
                    pass
            
            questions, error = self.extract_from_image(
                image_path=image_path,
                pdf_name=pdf_name,
                page_number=page_num
            )
            
            if questions:
                all_questions.extend(questions)
            
            if error:
                failed_pages.append(page_num)
                errors.append(error)
        
        return all_questions, failed_pages, errors