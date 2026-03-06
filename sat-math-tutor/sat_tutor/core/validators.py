"""
Data Validation Utilities
Used to validate if JSON output from LLM matches the expected schema
"""

import json
from typing import Type, TypeVar, Optional
from pydantic import BaseModel, ValidationError

from .models import Question, SolveResult, DiagnoseResult

T = TypeVar('T', bound=BaseModel)


class ValidationResult:
    """Validation result container"""
    def __init__(self, success: bool, data: Optional[BaseModel] = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error


def validate_json_to_model(json_str: str, model_class: Type[T]) -> ValidationResult:
    """
    Validates and converts a JSON string into a Pydantic model
    
    Args:
        json_str: The JSON string to validate
        model_class: The target Pydantic model class
    
    Returns:
        ValidationResult containing the success flag, data, or error message
    """
    try:
        # Attempt to parse JSON
        data = json.loads(json_str)
        # Validate and create model instance
        instance = model_class.model_validate(data)
        return ValidationResult(success=True, data=instance)
    except json.JSONDecodeError as e:
        return ValidationResult(success=False, error=f"JSON parsing failed: {str(e)}")
    except ValidationError as e:
        return ValidationResult(success=False, error=f"Schema validation failed: {str(e)}")


def validate_dict_to_model(data: dict, model_class: Type[T]) -> ValidationResult:
    """
    Validates and converts a dictionary into a Pydantic model
    """
    try:
        instance = model_class.model_validate(data)
        return ValidationResult(success=True, data=instance)
    except ValidationError as e:
        return ValidationResult(success=False, error=f"Schema validation failed: {str(e)}")


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extracts JSON content from text
    Handles markdown code blocks that may be output by the LLM
    """
    text = text.strip()
    
    # Try to find JSON code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    
    # Try to find generic code blocks
    if "```" in text:
        start = text.find("```") + 3
        # Skip potential language identifiers
        if text[start:start+10].strip().startswith('{') or text[start:start+10].strip().startswith('['):
            pass
        else:
            newline = text.find('\n', start)
            if newline != -1:
                start = newline + 1
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    
    # Try to find JSON objects or arrays directly
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start != -1:
            # Find the matching closing character
            depth = 0
            for i, c in enumerate(text[start:]):
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start:start+i+1]
    
    return None


def validate_questions_list(json_str: str) -> ValidationResult:
    """Validates a list of questions"""
    try:
        extracted = extract_json_from_text(json_str)
        if extracted is None:
            return ValidationResult(success=False, error="Could not extract JSON from text")
        
        data = json.loads(extracted)
        
        # Handle single question object or list of questions
        if isinstance(data, dict):
            if 'questions' in data:
                questions_data = data['questions']
            else:
                questions_data = [data]
        elif isinstance(data, list):
            questions_data = data
        else:
            return ValidationResult(success=False, error="Invalid JSON format: expected object or array")
        
        # Validate each question
        questions = []
        for q_data in questions_data:
            result = validate_dict_to_model(q_data, Question)
            if result.success:
                questions.append(result.data)
            else:
                return ValidationResult(success=False, error=f"Question validation failed: {result.error}")
        
        return ValidationResult(success=True, data=questions)
    except json.JSONDecodeError as e:
        return ValidationResult(success=False, error=f"JSON parsing failed: {str(e)}")


def validate_solve_result(json_str: str) -> ValidationResult:
    """Validates solving result"""
    extracted = extract_json_from_text(json_str)
    if extracted is None:
        return ValidationResult(success=False, error="Could not extract JSON from text")
    return validate_json_to_model(extracted, SolveResult)


def validate_diagnose_result(json_str: str) -> ValidationResult:
    """Validates diagnosis result"""
    extracted = extract_json_from_text(json_str)
    if extracted is None:
        return ValidationResult(success=False, error="Could not extract JSON from text")
    return validate_json_to_model(extracted, DiagnoseResult)