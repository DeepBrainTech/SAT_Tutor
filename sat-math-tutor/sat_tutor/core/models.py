"""
Core Data Model Definitions
Using Pydantic for strict data validation
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class QuestionSource(BaseModel):
    """Information regarding the source of the question"""
    pdf: str
    page: int


class UncertainSpan(BaseModel):
    """Annotation for uncertain content"""
    span: str = Field(..., description="The uncertain text content")
    reason: str = Field(..., description="Reason for uncertainty, e.g., blurry, cut-off, ambiguous")
    location: str = Field(..., description="Description of the location, e.g., choice C, stem")


class Question(BaseModel):
    """
    Question data model (Stage T output)
    Supports both SAT Math and SAT English problems
    """
    id: str = Field(..., description="Unique identifier for the question, format: p{page}_q{num}")
    source: QuestionSource
    exam: Literal["GRE", "SAT"] = "SAT"
    section: Literal["Math", "English"] = "Math"
    problem_type: Literal["multiple_choice", "numeric_entry", "unknown"] = "multiple_choice"
    stem: str = Field(..., description="The problem stem in plain text")
    choices: dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Options A-E (Math) or A-D (English), value is option content or null/UNKNOWN"
    )
    # Math-specific fields
    latex_equations: list[str] = Field(default_factory=list, description="LaTeX representation of equations")
    diagram_description: Optional[str] = Field(None, description="Textual description of diagrams/tables")
    constraints: list[str] = Field(default_factory=list, description="Constraints/conditions")
    # English-specific fields
    passage_context: Optional[str] = Field(None, description="Relevant passage text for reading questions")
    question_category: Optional[str] = Field(None, description="Question category: grammar, punctuation, vocabulary, etc.")
    # Common fields
    uncertain_spans: list[UncertainSpan] = Field(default_factory=list, description="Uncertainty annotations")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extraction confidence score 0-1")

    @field_validator('choices')
    @classmethod
    def validate_choices(cls, v: dict) -> dict:
        """Ensure option keys are restricted to A-E (Math) or A-D (English)"""
        valid_keys = {'A', 'B', 'C', 'D', 'E'}
        for key in v.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid choice key: {key}. Must be A-E.")
        return v


class SolveResult(BaseModel):
    """
    Solving result (Stage S output)
    """
    question_id: str
    correct_answer: str = Field(..., description="Correct answer (A-E or numeric value)")
    topic: str = Field(..., description="Topic classification: algebra, geometry, arithmetic, etc.")
    key_steps: list[str] = Field(..., min_length=1, max_length=10, description="3-7 key solving steps")
    final_reason: str = Field(..., description="One-sentence explanation of the final answer")
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator('correct_answer', mode='before')
    @classmethod
    def convert_answer_to_string(cls, v):
        """Converts numeric answers to string (as LLM might return a number instead of a string)"""
        if v is None:
            return ""
        return str(v)


class OptionAnalysis(BaseModel):
    """Analysis for an individual option"""
    option: str = Field(..., description="Option letter A-E")
    content: str = Field(..., description="Option text content")
    analysis: str = Field(..., description="Analytical explanation for this specific option")
    is_correct: bool = False
    is_user_choice: bool = False


class DiagnoseResult(BaseModel):
    """
    Error diagnosis result (Stage D output)
    """
    question_id: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    
    # Mode C scaffolded tutoring fields
    first_attempt: Optional[str] = Field(
        None,
        description="Student's first attempt answer (if Mode C scaffolded tutoring was used)"
    )
    first_attempt_wrong: bool = Field(
        False,
        description="True if the first attempt was wrong (even if second attempt was correct)"
    )
    student_work_image_path: Optional[str] = Field(
        None,
        description="Path to student's uploaded handwritten work image (if provided)"
    )
    student_work_transcription: Optional[str] = Field(
        None,
        description="LLM-transcribed student handwritten math work"
    )
    
    # Diagnosis content (populated only when the answer is incorrect)
    why_user_choice_is_tempting: Optional[str] = Field(
        None, 
        description="Explanation of why the user's choice might be a common pitfall"
    )
    likely_misconceptions: list[str] = Field(
        default_factory=list,
        description="Potential cognitive misconceptions, at least 2 items"
    )
    how_to_get_correct: Optional[str] = Field(
        None,
        description="Corrective path + correct solving steps"
    )
    option_analysis: list[OptionAnalysis] = Field(
        default_factory=list,
        description="Option analysis, covering at least user_answer and correct_answer"
    )


class TranscribeOutput(BaseModel):
    """Complete output for Stage T"""
    questions: list[Question]
    total_pages: int
    failed_pages: list[int] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SessionResult(BaseModel):
    """Complete session results"""
    session_id: str
    pdf_path: str
    mode: str
    timestamp: str
    
    # Stage T
    transcribed: TranscribeOutput
    
    # Stage S (Optional)
    solve_results: list[SolveResult] = Field(default_factory=list)
    
    # Stage D (Optional)
    diagnose_results: list[DiagnoseResult] = Field(default_factory=list)
    
    # Summary Statistics
    total_questions: int = 0
    answered_questions: int = 0
    correct_count: int = 0
    
    # Mode C scaffolded tutoring statistics
    first_attempt_wrong_count: int = Field(
        0, 
        description="Number of questions where first attempt was wrong (Mode C)"
    )
    first_attempt_wrong_ids: list[str] = Field(
        default_factory=list,
        description="Question IDs where first attempt was wrong"
    )
    incorrect_ids: list[str] = Field(default_factory=list)
