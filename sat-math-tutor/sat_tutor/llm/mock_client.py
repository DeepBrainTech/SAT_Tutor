"""
Mock LLM Client
Used for testing and demonstration when no API Key is available
"""

import json
import random
from typing import Optional

from .base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    """Mock LLM Client that returns preset test data"""
    
    def __init__(self):
        self._question_counter = 0
    
    @property
    def is_available(self) -> bool:
        return True
    
    def _generate_mock_question(self, page: int) -> dict:
        """Generates a mock question"""
        self._question_counter += 1
        
        mock_questions = [
            {
                "stem": "If x + 5 = 12, what is the value of x?",
                "choices": {"A": "5", "B": "6", "C": "7", "D": "8", "E": "17"},
                "latex_equations": ["x + 5 = 12"],
                "correct_answer": "C",
                "topic": "algebra"
            },
            {
                "stem": "What is the area of a circle with radius 4?",
                "choices": {"A": "4π", "B": "8π", "C": "12π", "D": "16π", "E": "64π"},
                "latex_equations": ["A = \\pi r^2"],
                "correct_answer": "D",
                "topic": "geometry"
            },
            {
                "stem": "If 3x - 7 = 14, what is the value of x?",
                "choices": {"A": "3", "B": "5", "C": "7", "D": "9", "E": "21"},
                "latex_equations": ["3x - 7 = 14"],
                "correct_answer": "C",
                "topic": "algebra"
            },
            {
                "stem": "What is 25% of 80?",
                "choices": {"A": "15", "B": "20", "C": "25", "D": "30", "E": "40"},
                "latex_equations": [],
                "correct_answer": "B",
                "topic": "arithmetic"
            },
            {
                "stem": "In a right triangle, if one leg is 3 and another leg is 4, what is the hypotenuse?",
                "choices": {"A": "4", "B": "5", "C": "6", "D": "7", "E": "12"},
                "latex_equations": ["a^2 + b^2 = c^2"],
                "correct_answer": "B",
                "topic": "geometry"
            }
        ]
        
        idx = (self._question_counter - 1) % len(mock_questions)
        q = mock_questions[idx]
        
        return {
            "id": f"p{page}_q{self._question_counter}",
            "source": {"pdf": "mock.pdf", "page": page},
            "exam": "SAT",
            "section": "Math",
            "problem_type": "multiple_choice",
            "stem": q["stem"],
            "choices": q["choices"],
            "latex_equations": q["latex_equations"],
            "diagram_description": None,
            "constraints": [],
            "uncertain_spans": [],
            "confidence": 0.95,
            "_mock_correct": q["correct_answer"],
            "_mock_topic": q["topic"]
        }
    
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: Optional[str] = None,
        images: Optional[list[str]] = None,
        temperature: float = 0.1
    ) -> LLMResponse:
        """Generates a mock JSON response"""
        
        # Determine the type of response based on prompt content
        prompt_lower = (system_prompt + user_prompt).lower()
        
        if any(keyword in prompt_lower for keyword in ["transcribe", "抽取", "转写"]):
            # Stage T: Extraction
            page = 1
            if images:
                # Extract page number from image path
                import re
                for img in images:
                    match = re.search(r'page_(\d+)', img)
                    if match:
                        page = int(match.group(1))
                        break
            
            question = self._generate_mock_question(page)
            # Remove mock-specific internal fields
            question.pop('_mock_correct', None)
            question.pop('_mock_topic', None)
            
            result = {"questions": [question]}
            
        elif any(keyword in prompt_lower for keyword in ["solve", "求解"]):
            # Stage S: Solving
            # Extract question ID from prompt
            import re
            q_id_match = re.search(r'(p\d+_q\d+)', user_prompt)
            q_id = q_id_match.group(1) if q_id_match else "p1_q1"
            
            result = {
                "question_id": q_id,
                "correct_answer": "C",
                "topic": "algebra",
                "key_steps": [
                    "Identify problem type: Linear equation in one variable",
                    "Rearrange and simplify the equation",
                    "Solve for the unknown value x",
                    "Verify the correctness of the answer"
                ],
                "final_reason": "Correct answer obtained through algebraic manipulation of the equation.",
                "confidence": 0.92
            }
            
        elif any(keyword in prompt_lower for keyword in ["diagnose", "诊断", "错因"]):
            # Stage D: Diagnosis
            import re
            q_id_match = re.search(r'(p\d+_q\d+)', user_prompt)
            q_id = q_id_match.group(1) if q_id_match else "p1_q1"
            
            # Check if it is numeric entry or multiple choice (based on prompt content)
            is_numeric_entry = any(keyword in prompt_lower for keyword in ["填空题", "numeric"])
            
            if is_numeric_entry:
                # Numeric Entry Diagnosis
                user_ans_match = re.search(r'user[_\s]?answer[^:]*:\s*([^\n]+)', user_prompt, re.I)
                user_ans = user_ans_match.group(1).strip() if user_ans_match else "10"
                
                correct_ans_match = re.search(r'correct[_\s]?answer[^:]*:\s*([^\n]+)', user_prompt, re.I)
                correct_ans = correct_ans_match.group(1).strip() if correct_ans_match else "12"
                
                is_correct = user_ans == correct_ans
                
                result = {
                    "question_id": q_id,
                    "user_answer": user_ans,
                    "correct_answer": correct_ans,
                    "is_correct": is_correct,
                    "why_user_answer_is_wrong": None if is_correct else f"Your answer is {user_ans}, but the correct answer is {correct_ans}. An error might have occurred during the calculation, such as forgetting a coefficient or mishandling a sign.",
                    "likely_misconceptions": [] if is_correct else [
                        "A step might have been missed during the calculation",
                        "Possible misunderstanding in the application of the formula"
                    ],
                    "how_to_get_correct": None if is_correct else f"To get the correct answer {correct_ans}, you need to:\n1. Read the problem carefully to clarify what is required\n2. List the correct calculation steps\n3. Calculate step-by-step and verify the result",
                    "error_type": "Calculation Error"
                }
            else:
                # Multiple Choice Diagnosis
                user_ans_match = re.search(r'user[_\s]?answer[^:]*:\s*([A-E])', user_prompt, re.I)
                user_ans = user_ans_match.group(1).upper() if user_ans_match else "A"
                
                correct_ans_match = re.search(r'correct[_\s]?answer[^:]*:\s*([A-E])', user_prompt, re.I)
                correct_ans = correct_ans_match.group(1).upper() if correct_ans_match else "C"
                
                is_correct = user_ans == correct_ans
                
                result = {
                    "question_id": q_id,
                    "user_answer": user_ans,
                    "correct_answer": correct_ans,
                    "is_correct": is_correct,
                    "why_user_choice_is_tempting": None if is_correct else f"Option {user_ans} might have been chosen due to an intermediate calculation result or a misreading of the problem conditions. This is a common distractor.",
                    "likely_misconceptions": [] if is_correct else [
                        "Possible confusion between similar formulas or concepts",
                        "Possible sign or numerical error during calculation"
                    ],
                    "how_to_get_correct": None if is_correct else f"To get the correct answer {correct_ans}, you need to:\n1. Read the question carefully to identify given conditions\n2. Choose the correct formula or method\n3. Perform accurate calculations\n4. Verify if the answer is reasonable",
                    "option_analysis": [
                        {
                            "option": user_ans,
                            "content": f"Option {user_ans}",
                            "analysis": "The option selected by the user" + (", correct" if is_correct else ", which is a distractor"),
                            "is_correct": is_correct,
                            "is_user_choice": True
                        },
                        {
                            "option": correct_ans,
                            "content": f"Option {correct_ans}",
                            "analysis": "The correct answer, obtained through the proper method",
                            "is_correct": True,
                            "is_user_choice": user_ans == correct_ans
                        }
                    ]
                }
        else:
            # Return empty object by default
            result = {}
        
        return LLMResponse(
            content=json.dumps(result, ensure_ascii=False, indent=2),
            success=True
        )
    
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3
    ) -> LLMResponse:
        """Generates a mock text response"""
        return LLMResponse(
            content="This is a text response generated in Mock mode. Please configure OPENAI_API_KEY for real responses.",
            success=True
        )