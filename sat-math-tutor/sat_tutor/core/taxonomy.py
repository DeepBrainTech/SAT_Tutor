"""
SAT Math Error Classification System
Taxonomy used for misconception classification during the diagnosis stage
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class MisconceptionType:
    """Misconception type definition"""
    code: str
    name: str
    description: str
    examples: list[str]


# Common SAT Math Misconception Taxonomy
MISCONCEPTION_TAXONOMY = {
    # Calculation Errors
    "CALC_SIGN": MisconceptionType(
        code="CALC_SIGN",
        name="Sign Error",
        description="Incorrect handling of positive/negative signs, such as changing subtraction to addition",
        examples=["-(-3) miscalculated as -3", "Forgetting to change the sign when moving terms across an equation"]
    ),
    "CALC_ORDER": MisconceptionType(
        code="CALC_ORDER",
        name="Order of Operations Error",
        description="Failing to calculate according to the correct operator precedence (PEMDAS/BODMAS)",
        examples=["2+3×4 calculated as 20 instead of 14", "Performing addition/subtraction before multiplication/division"]
    ),
    "CALC_DECIMAL": MisconceptionType(
        code="CALC_DECIMAL",
        name="Decimal/Fraction Calculation Error",
        description="Mistakes in decimal point placement or fraction operations",
        examples=["0.1 × 0.1 = 0.01 miscalculated as 0.1", "Errors in finding a common denominator"]
    ),
    "CALC_POWER": MisconceptionType(
        code="CALC_POWER",
        name="Exponent/Power Error",
        description="Incorrect application of exponent rules",
        examples=["(x²)³ miscalculated as x⁵", "x² × x³ miscalculated as x⁶"]
    ),
    
    # Conceptual Errors
    "CONCEPT_FORMULA": MisconceptionType(
        code="CONCEPT_FORMULA",
        name="Formula Misremembrance",
        description="Remembering or confusing related formulas incorrectly",
        examples=["Using 2πr instead of πr² for the area of a circle", "Confusing Permutation and Combination formulas"]
    ),
    "CONCEPT_DEFINITION": MisconceptionType(
        code="CONCEPT_DEFINITION",
        name="Definitional Misunderstanding",
        description="Inaccurate understanding of the definition of mathematical concepts",
        examples=["Misunderstanding the definition of 'median'", "Confusing 'factors' with 'multiples'"]
    ),
    "CONCEPT_PROPERTY": MisconceptionType(
        code="CONCEPT_PROPERTY",
        name="Property Misapplication",
        description="Error in understanding or applying mathematical properties",
        examples=["Assuming the square of a negative number is still negative", "Incorrect application of parallel line properties"]
    ),
    
    # Reading/Interpretation Errors
    "READ_CONDITION": MisconceptionType(
        code="READ_CONDITION",
        name="Missing Condition",
        description="Ignoring important conditions or constraints in the problem",
        examples=["Ignoring the x > 0 constraint", "Overlooking the 'integer' requirement"]
    ),
    "READ_QUESTION": MisconceptionType(
        code="READ_QUESTION",
        name="Misreading the Question",
        description="Deviation in understanding what the question is asking for",
        examples=["Calculating perimeter when asked for area", "Finding the maximum when asked for the minimum"]
    ),
    "READ_UNITS": MisconceptionType(
        code="READ_UNITS",
        name="Unit Conversion Error",
        description="Ignoring or incorrectly processing unit conversions",
        examples=["Mixing minutes and hours", "Failing to convert centimeters to meters"]
    ),
    
    # Trap/Distractor Errors
    "TRAP_INTERMEDIATE": MisconceptionType(
        code="TRAP_INTERMEDIATE",
        name="Intermediate Result Trap",
        description="Mistaking an intermediate calculation step for the final answer",
        examples=["Solving for x and picking that answer when the question asks for 2x", "Picking a result before final simplification"]
    ),
    "TRAP_SIMILAR": MisconceptionType(
        code="TRAP_SIMILAR",
        name="Similar Value Trap",
        description="Choosing a distractor that is numerically similar to the correct answer",
        examples=["Picking 21 when the correct answer is 12", "Picking 4/3 when the correct answer is 3/4"]
    ),
    "TRAP_COMMON": MisconceptionType(
        code="TRAP_COMMON",
        name="Common Error Trap",
        description="Options specifically designed to match results of common mistakes",
        examples=["A choice that matches a sign error result", "A choice that matches a result missing the final step"]
    ),
    
    # Methodological Errors
    "METHOD_WRONG": MisconceptionType(
        code="METHOD_WRONG",
        name="Incorrect Method Selection",
        description="Using a solution method that is not applicable to the problem",
        examples=["Using arithmetic when an algebraic equation is required", "Forgetting to flip the inequality sign when dividing by a negative number"]
    ),
    "METHOD_INCOMPLETE": MisconceptionType(
        code="METHOD_INCOMPLETE",
        name="Incomplete Solution",
        description="Missing steps or failing to complete the solution process",
        examples=["Solving an equation for only one of its roots", "Forgetting to check and discard extraneous roots"]
    ),
    
    # Logical Errors
    "LOGIC_REVERSE": MisconceptionType(
        code="LOGIC_REVERSE",
        name="Reverse Logic",
        description="Reversing the relationship between conditions and conclusions",
        examples=["Confusing sufficient and necessary conditions", "Treating the converse as the original proposition"]
    ),
    "LOGIC_SCOPE": MisconceptionType(
        code="LOGIC_SCOPE",
        name="Scope/Range Error",
        description="Incorrect judgment regarding numerical ranges or set relationships",
        examples=["Incorrectly determining the solution set of an inequality", "Confusing Union and Intersection of sets"]
    )
}


def get_misconception_by_code(code: str) -> Optional[MisconceptionType]:
    """Retrieves a misconception type by its code"""
    return MISCONCEPTION_TAXONOMY.get(code)


def get_misconceptions_by_topic(topic: str) -> list[MisconceptionType]:
    """Retrieves likely misconception types based on the problem topic"""
    topic_misconceptions = {
        "algebra": ["CALC_SIGN", "CALC_ORDER", "CONCEPT_FORMULA", "METHOD_WRONG", "READ_CONDITION"],
        "geometry": ["CONCEPT_FORMULA", "CONCEPT_PROPERTY", "READ_UNITS", "CALC_DECIMAL"],
        "arithmetic": ["CALC_ORDER", "CALC_DECIMAL", "CALC_POWER", "READ_UNITS"],
        "data_analysis": ["CONCEPT_DEFINITION", "READ_QUESTION", "CALC_DECIMAL"],
        "number_theory": ["CONCEPT_DEFINITION", "CONCEPT_PROPERTY", "READ_CONDITION"],
        "word_problems": ["READ_CONDITION", "READ_QUESTION", "READ_UNITS", "METHOD_WRONG"]
    }
    
    codes = topic_misconceptions.get(topic, list(MISCONCEPTION_TAXONOMY.keys())[:5])
    return [MISCONCEPTION_TAXONOMY[code] for code in codes if code in MISCONCEPTION_TAXONOMY]


def format_misconception_for_prompt(misconception: MisconceptionType) -> str:
    """Formats a misconception type for use in a prompt"""
    return f"- {misconception.name}: {misconception.description}"


def get_all_misconceptions_prompt() -> str:
    """Retrieves all misconception types in a formatted text for prompts"""
    lines = ["Common SAT Math Misconception Categories:"]
    for m in MISCONCEPTION_TAXONOMY.values():
        lines.append(format_misconception_for_prompt(m))
    return "\n".join(lines)