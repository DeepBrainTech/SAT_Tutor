"""
Student Simulation Module
Use LLM to simulate student answering, intentionally making some mistakes

Configuration:
--------------
Configure student model via environment variables (supports DeepSeek, OpenAI and compatible APIs):

  # Student simulation dedicated API config (optional, uses main API if not set)
  STUDENT_API_KEY=sk-xxx              # Student model API Key
  STUDENT_API_BASE=https://api.deepseek.com  # API Base URL
  STUDENT_MODEL=deepseek-chat         # Model name
  STUDENT_CORRECT_RATE=70             # Accuracy rate (integer)

Common API Base URLs:
  - OpenAI: https://api.openai.com/v1 (default)
  - DeepSeek: https://api.deepseek.com
  - Local Ollama: http://localhost:11434/v1
"""

import json
import os
from typing import Optional
import random
from ..core.models import Question, SolveResult
from ..llm.base import LLMClient


def get_student_config() -> dict:
    """
    Get student model configuration
    
    Returns:
        Config dict containing api_key, api_base, model and correct_rate
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    # Student model API config
    # Prefer STUDENT_* dedicated config, otherwise use OPENAI_* main config
    student_api_key = os.getenv("STUDENT_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    student_api_base = os.getenv("STUDENT_API_BASE") or os.getenv("OPENAI_API_BASE")
    student_model = os.getenv("STUDENT_MODEL") or os.getenv("OPENAI_MODEL_STUDENT") or os.getenv("OPENAI_MODEL_TEXT", "gpt-4o-mini")
    
    # Accuracy rate (specific number, e.g. 70 means 70%)
    correct_rate_str = os.getenv("STUDENT_CORRECT_RATE", "70")
    try:
        correct_rate = int(correct_rate_str)
    except:
        correct_rate = 70
    
    return {
        "api_key": student_api_key,
        "api_base": student_api_base,
        "model": student_model,
        "correct_rate": correct_rate
    }


def create_student_llm_client(config: dict = None) -> LLMClient:
    """
    Create dedicated LLM client for student simulation
    
    Args:
        config: Config dict, reads from env if None
    
    Returns:
        LLMClient instance
    """
    if config is None:
        config = get_student_config()
    
    from ..llm.openai_client import OpenAIClient
    
    client = OpenAIClient(
        api_key=config.get("api_key"),
        text_model=config.get("model"),
        api_base=config.get("api_base")
    )
    
    return client


def get_student_system_prompt(correct_rate: int = 70) -> str:
    """
    Generates the system prompt for student simulation.
    Control of specific errors is now handled via the User Prompt instructions.
    """
    return f"""You are an AI simulating a student taking a math test. Your goal is to act like a real student learning SAT math.
Your target overall accuracy is approximately {correct_rate}%.

[IMPORTANT: Reasoning-First Logic]
For every question, you MUST generate JSON fields in this exact physical order:
1. "thought_process": 
   - First, check if this Question ID is on your "Intended Error List".
   - If answering CORRECTLY: Write a clear step-by-step derivation. Double-check that your numerical result matches the text of the option letter you choose.
   - If answering WRONG: Pick a realistic human error (e.g., misreading a sign, careless arithmetic, forgetting a constraint, or picking an intermediate result). Describe this flawed logic.
   - Conclusion requirement: End this field with "My result is [X], which corresponds to option [Y]".
2. "made_mistake": A boolean indicating if you intentionally introduced an error in the thought process.
3. "answer": The final answer based strictly on the thought process above.

[CRITICAL ANSWER FORMAT RULES - MUST FOLLOW]
- For "multiple_choice" questions: answer MUST be EXACTLY ONE LETTER from A, B, C, D, E
  * WRONG: "24", "14", "1/2", "answer is A"
  * CORRECT: "A", "B", "C", "D", "E"
- For "numeric_entry" questions: answer MUST be the NUMBER
  * WRONG: "A", "option B"
  * CORRECT: "24", "14", "0.5", "-3"

[Strict Constraints]
- NEVER provide the "answer" before completing the "thought_process".
- For multiple choice: Find which OPTION LETTER contains your calculated result, then output ONLY that letter.
- Mistakes must look like "human slips," not random gibberish.

Output only JSON. No conversational filler."""


# Keep a default for backward compatibility
STUDENT_SIMULATOR_SYSTEM_PROMPT = get_student_system_prompt(70)

STUDENT_SIMULATOR_USER_PROMPT_TEMPLATE = """Please simulate a student answering the following {total} questions:

{questions_text}

[STRICT ACCURACY EXECUTION]
- Total Questions: {total}
- Target Accuracy: {correct_rate}%
- 【MANDATORY】You must INTENTIONALLY ANSWER WRONG on these specific Question IDs: {error_ids}
- 【MANDATORY】For all other IDs, you must provide the CORRECT answer.

[ANSWER FORMAT - EXTREMELY IMPORTANT]
- For questions marked "multiple_choice": Your "answer" field MUST be a single letter: A, B, C, D, or E
- For questions marked "numeric_entry": Your "answer" field MUST be a number like "24" or "14"

Output JSON format."""

STUDENT_SCHEMA_HINT = """{
  "p1_q1": {
    "thought_process": "Step by step reasoning...",
    "made_mistake": false,
    "answer": "A"
  },
  "p1_q2": {
    "thought_process": "Step by step reasoning...",
    "made_mistake": true,
    "answer": "14"
  }
}

IMPORTANT: 
- Each key must be the question_id (e.g. "p1_q1", "p1_q2")
- Each value must contain: thought_process, made_mistake, answer
- Output ALL questions, not just one!"""


# ============================================================
# SAT English - Student Simulation Prompts
# ============================================================

def get_english_student_system_prompt(correct_rate: int = 70) -> str:
    """
    Generates the system prompt for English student simulation.
    Simulates common mistakes in grammar, reading comprehension, etc.
    """
    return f"""You are an AI simulating a student taking the SAT English test. Your goal is to act like a real student learning English grammar and reading comprehension.
Your target overall accuracy is approximately {correct_rate}%.

[IMPORTANT: Reasoning-First Logic]
For every question, you MUST generate JSON fields in this exact order:
1. "thought_process": 
   - First, check if this Question ID is on your "Intended Error List".
   - If answering CORRECTLY: Explain your understanding of the grammar rule or reading passage, then select the correct option.
   - If answering WRONG: Make a realistic student mistake such as:
     * Grammar: Misunderstanding subject-verb agreement, pronoun reference, or tense
     * Punctuation: Misusing commas, semicolons, or apostrophes
     * Reading: Misinterpreting the passage, missing context clues, or choosing a distractor
     * Vocabulary: Confusing similar words or not understanding context
   - End with "Based on my analysis, I choose option [X]".
2. "made_mistake": A boolean indicating if you intentionally made an error.
3. "answer": The final answer based on your thought process.

[CRITICAL ANSWER FORMAT RULES]
- SAT English has 4 options: A, B, C, D
- answer MUST be EXACTLY ONE LETTER: A, B, C, or D
- WRONG: "Option A", "A is correct", "the answer is A"
- CORRECT: "A", "B", "C", "D"

[Common Student Mistakes to Simulate]
- Grammar: Choosing wordy/awkward phrasing, ignoring parallel structure
- Punctuation: Missing comma splices, misusing semicolons
- Reading: Selecting answers that are too extreme or not supported by text
- Transitions: Choosing transitions that don't match the logical flow
- Vocabulary: Picking words that sound right but don't fit the context

[Strict Constraints]
- Complete "thought_process" BEFORE giving "answer"
- Mistakes must look like genuine student errors, not random guessing
- Always select exactly one letter (A/B/C/D)

Output only JSON. No conversational filler."""


ENGLISH_STUDENT_USER_PROMPT_TEMPLATE = """Please simulate a student answering the following {total} SAT English questions:

{questions_text}

[STRICT ACCURACY EXECUTION]
- Total Questions: {total}
- Target Accuracy: {correct_rate}%
- 【MANDATORY】You must INTENTIONALLY ANSWER WRONG on these specific Question IDs: {error_ids}
- 【MANDATORY】For all other IDs, you must provide the CORRECT answer.

[ANSWER FORMAT - SAT English]
- ALL answers must be a single letter: A, B, C, or D
- NO numeric answers for English questions
- Think through each question as a real student would

Output JSON format."""


ENGLISH_STUDENT_SCHEMA_HINT = """{
  "p1_q1": {
    "thought_process": "This question is about subject-verb agreement. The subject is 'team' which is singular, so I need a singular verb. Option A uses 'are' which is plural. Option B uses 'is' which matches. Based on my analysis, I choose option B.",
    "made_mistake": false,
    "answer": "B"
  },
  "p1_q2": {
    "thought_process": "This looks like a punctuation question. I think both parts can stand alone, so maybe I need a semicolon... but actually commas look fine here. Based on my analysis, I choose option A.",
    "made_mistake": true,
    "answer": "A"
  }
}

IMPORTANT:
- Each key must be the question_id (e.g. "p1_q1", "p1_q2")
- Each value must contain: thought_process, made_mistake, answer
- answer must be A, B, C, or D (not numbers!)
- Output ALL questions!"""


def format_questions_for_simulator(
    questions: list[Question], 
    solve_results: Optional[list[SolveResult]] = None,
    subject: str = "math"
) -> str:
    """
    Format questions for simulator use
    
    Args:
        questions: List of questions
        solve_results: Optional solve results
        subject: "math" or "english"
    """
    solve_map = {sr.question_id: sr for sr in (solve_results or [])}
    
    lines = []
    for i, q in enumerate(questions, 1):
        lines.append(f"\n--- Question {i}: {q.id} ---")
        
        if subject == "english":
            # English questions - always multiple choice A-D
            lines.append(f"Type: multiple_choice")
            lines.append(f">>> ANSWER FORMAT: Must be ONE letter from A/B/C/D <<<")
            
            # Add question category if available
            if hasattr(q, 'question_category') and q.question_category:
                lines.append(f"Category: {q.question_category}")
            
            lines.append(f"Question: {q.stem}")
            
            # Add passage context if available
            if hasattr(q, 'passage_context') and q.passage_context:
                lines.append(f"Passage Context: {q.passage_context}")
            
            if q.choices:
                lines.append("Options:")
                for opt in ['A', 'B', 'C', 'D']:
                    content = q.choices.get(opt)
                    if content and content not in ["N/A", "UNKNOWN", None, ""]:
                        lines.append(f"  {opt}: {content}")
        else:
            # Math questions
            is_multiple_choice = q.problem_type != "numeric_entry"
            
            if is_multiple_choice:
                lines.append(f"Type: multiple_choice")
                lines.append(f">>> ANSWER FORMAT: Must be ONE letter from A/B/C/D/E <<<")
            else:
                lines.append(f"Type: numeric_entry")
                lines.append(f">>> ANSWER FORMAT: Must be a NUMBER <<<")
            
            lines.append(f"Stem: {q.stem}")
            
            # Include LaTeX equations if available
            if q.latex_equations:
                lines.append(f"Formulas: {', '.join(q.latex_equations)}")
            
            # Include diagram description if available
            if q.diagram_description:
                lines.append(f"Diagram: {q.diagram_description}")
            
            if is_multiple_choice and q.choices:
                lines.append("Options:")
                for opt in ['A', 'B', 'C', 'D', 'E']:
                    content = q.choices.get(opt)
                    if content and content not in ["N/A", "UNKNOWN", None, ""]:
                        lines.append(f"  {opt}: {content}")
    
    return "\n".join(lines)


def validate_and_fix_answer(answer: str, question: Question) -> str:
    """
    Validate and fix answer format based on question type.
    For multiple choice: ensure answer is A-E, try to map numeric answers to options.
    For numeric entry: ensure answer is a number.
    
    Args:
        answer: The raw answer from LLM
        question: The question object
    
    Returns:
        Validated/fixed answer
    """
    answer = str(answer).strip().upper()
    is_multiple_choice = question.problem_type != "numeric_entry"
    
    if is_multiple_choice:
        # Already a valid option letter
        if answer in ['A', 'B', 'C', 'D', 'E']:
            return answer
        
        # Try to extract letter if answer contains it
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if f"OPTION {letter}" in answer or f"({letter})" in answer:
                return letter
        
        # Try to map numeric answer to option
        if question.choices:
            answer_lower = answer.lower().strip()
            for opt, content in question.choices.items():
                if content:
                    content_clean = str(content).strip().lower()
                    # Exact match
                    if answer_lower == content_clean:
                        return opt
                    # Numeric match (handle "24" matching "24" or "$24")
                    try:
                        ans_num = float(answer_lower.replace(',', '').replace('$', ''))
                        content_num = float(content_clean.replace(',', '').replace('$', ''))
                        if abs(ans_num - content_num) < 0.001:
                            return opt
                    except:
                        pass
        
        # Last resort: return first letter if answer looks like a number
        # This indicates the LLM gave a calculated value instead of option
        try:
            float(answer.replace(',', ''))
            # It's a number, but we couldn't map it - return as-is with warning
            return answer  # Will be caught in diagnosis as wrong format
        except:
            pass
        
        return answer
    else:
        # Numeric entry - remove any option letters
        if answer in ['A', 'B', 'C', 'D', 'E'] and question.choices:
            # LLM gave option letter for numeric question, try to get the value
            content = question.choices.get(answer, "")
            if content:
                return str(content).strip()
        return answer


def simulate_student_answers(
    llm_client: LLMClient,
    questions: list[Question],
    solve_results: Optional[list[SolveResult]] = None,
    correct_rate: int = 70,
    subject: str = "math"
) -> tuple[dict[str, str], dict]:
    """
    Use LLM to simulate student answering
    
    Args:
        llm_client: LLM client
        questions: Question list
        solve_results: Correct answers (optional, not directly used by simulator)
        correct_rate: Accuracy rate (0-100 integer)
        subject: "math" or "english" - determines which prompts to use
    
    Returns:
        Tuple of (answers dict {question_id: answer}, full_details dict)
    """
    total = len(questions)
    correct_count = round(total * correct_rate / 100)
    error_count = total - correct_count
    
    question_ids = [q.id for q in questions]
    # Randomly sample which IDs will be "Intentionally Wrong"
    error_ids = random.sample(question_ids, min(error_count, len(question_ids)))
    
    # Step 2: Prepare the prompts based on subject
    questions_text = format_questions_for_simulator(questions, solve_results, subject)
    
    if subject == "english":
        # Use English-specific prompts
        system_prompt = get_english_student_system_prompt(correct_rate)
        user_prompt = ENGLISH_STUDENT_USER_PROMPT_TEMPLATE.format(
            total=total,
            correct_rate=correct_rate,
            error_ids=error_ids,
            questions_text=questions_text
        )
        schema_hint = ENGLISH_STUDENT_SCHEMA_HINT
    else:
        # Use Math prompts (default)
        system_prompt = get_student_system_prompt(correct_rate)
        user_prompt = STUDENT_SIMULATOR_USER_PROMPT_TEMPLATE.format(
            total=total,
            correct_rate=correct_rate,
            error_ids=error_ids,
            questions_text=questions_text
        )
        schema_hint = STUDENT_SCHEMA_HINT
    
    # Step 3: Call LLM with a slightly higher temperature for "natural" human errors
    response = llm_client.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_hint=schema_hint,
        temperature=0.7 
    )
    
    if not response.success:
        raise ValueError(f"Student simulation failed: {response.error}")
    
    try:
        content = response.content.strip()
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        data = json.loads(content)
        
        answers = {}
        full_details = {}
        
        # Handle different response formats
        if "answers" in data:
            # Format: {"answers": {"p1_q1": {...}, ...}}
            for q_id, info in data["answers"].items():
                if isinstance(info, dict):
                    answers[q_id] = str(info.get("answer", ""))
                    full_details[q_id] = info
                else:
                    answers[q_id] = str(info)
                    full_details[q_id] = {"answer": str(info)}
        else:
            # Format: {"p1_q1": {...}, "p1_q2": {...}, ...}
            for k, v in data.items():
                # Skip metadata keys
                if k.startswith("_") or k in ["summary", "total", "thought_process", "made_mistake", "answer"]:
                    continue
                # Only process keys that look like question IDs
                if k.startswith("p") and "_q" in k:
                    if isinstance(v, dict):
                        answers[k] = str(v.get("answer", ""))
                        full_details[k] = v
                    else:
                        answers[k] = str(v)
                        full_details[k] = {"answer": str(v)}
        
        # Fallback: if no valid question IDs found, try to map by order
        if not answers and len(questions) == 1:
            # Single question case - the response might be the answer directly
            if "answer" in data:
                q_id = questions[0].id
                answers[q_id] = str(data.get("answer", ""))
                full_details[q_id] = data
        
        # Step 4: Validate and fix answers based on question type
        question_map = {q.id: q for q in questions}
        validated_answers = {}
        for q_id, raw_answer in answers.items():
            if q_id in question_map:
                validated_answer = validate_and_fix_answer(raw_answer, question_map[q_id])
                validated_answers[q_id] = validated_answer
                # Update full_details if answer was changed
                if validated_answer != raw_answer and q_id in full_details:
                    full_details[q_id]["original_answer"] = raw_answer
                    full_details[q_id]["answer"] = validated_answer
            else:
                validated_answers[q_id] = raw_answer
        
        return validated_answers, full_details
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse simulated answers: {e}\nRaw response: {response.content[:500]}")


def simulate_and_save_answers(
    llm_client: LLMClient,
    questions: list[Question],
    output_path: str,
    solve_results: Optional[list[SolveResult]] = None,
    correct_rate: int = 70,
    subject: str = "math"
) -> dict[str, str]:
    """
    Simulate student answering and save to file
    
    Args:
        llm_client: LLM client
        questions: Question list
        output_path: Output file path
        solve_results: Correct answers (optional)
        correct_rate: Accuracy rate (0-100 integer)
        subject: "math" or "english"
    
    Returns:
        Simulated student answers dict
    """
    from rich.console import Console
    console = Console()
    
    total = len(questions)
    correct_count = round(total * correct_rate / 100)
    error_count = total - correct_count
    
    subject_name = "SAT English" if subject == "english" else "Math"
    console.print(f"\n[bold magenta]Simulating student answering ({subject_name})...[/bold magenta]")
    console.print(f"[dim]Total {total} questions, target {correct_count} correct, {error_count} wrong (accuracy {correct_rate}%)[/dim]\n")
    
    answers, full_details = simulate_student_answers(
        llm_client, questions, solve_results,
        correct_rate=correct_rate,
        subject=subject
    )
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save simple answers format (for diagnose stage)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    
    # Save full details to a separate file (for debugging)
    details_path = output_path.replace('.json', '_details.json')
    with open(details_path, 'w', encoding='utf-8') as f:
        json.dump(full_details, f, ensure_ascii=False, indent=2)
    
    console.print(f"[green]Done! Generated {len(answers)} answers[/green]")
    console.print(f"[green]Saved to: {output_path}[/green]")
    console.print(f"[dim]Details saved to: {details_path}[/dim]")
    
    console.print("\n[bold]Answer preview:[/bold]")
    for q_id, ans in list(answers.items())[:3]:
        detail = full_details.get(q_id, {})
        made_mistake = detail.get("made_mistake", "?")
        console.print(f"  [cyan]{q_id}[/cyan]: [yellow]{ans}[/yellow] (mistake: {made_mistake})")
    if len(answers) > 3:
        console.print(f"  ... total {len(answers)} answers")
    
    return answers


def ask_simulate_student(
    llm_client: LLMClient,
    questions: list[Question],
    solve_results: Optional[list[SolveResult]] = None,
    session_dir: Optional[str] = None,
    subject: str = "math"
) -> Optional[dict[str, str]]:
    """
    Ask user if they want to simulate student answering
    
    Args:
        llm_client: LLM client
        questions: List of questions
        solve_results: Optional solve results
        session_dir: Session directory for saving output
        subject: "math" or "english" - determines which prompts to use
    
    Returns:
        Simulated answers dict, or None if user cancels
    """
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    
    config = get_student_config()
    
    subject_name = "SAT English" if subject == "english" else "Math"
    console.print("\n" + "="*70, style="magenta")
    console.print(f"Simulate Student Answering ({subject_name})", style="bold magenta")
    console.print("="*70, style="magenta")
    
    console.print("\n[bold]Current Config:[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_column("Env Variable", style="dim")
    
    api_key_display = config["api_key"][:8] + "..." + config["api_key"][-4:] if len(config["api_key"]) > 12 else ("Set" if config["api_key"] else "Not Set")
    api_base_display = config["api_base"] or "Default (OpenAI)"
    
    table.add_row("API Key", api_key_display, "STUDENT_API_KEY")
    table.add_row("API Base", api_base_display, "STUDENT_API_BASE")
    table.add_row("Model", config["model"], "STUDENT_MODEL")
    table.add_row("Accuracy", f"{config['correct_rate']}%", "STUDENT_CORRECT_RATE")
    console.print(table)
    
    console.print(Panel("""[dim]Common API config examples (.env file):

# OpenAI (default)
STUDENT_API_KEY=sk-xxx
STUDENT_MODEL=gpt-4o-mini

# DeepSeek
STUDENT_API_KEY=sk-xxx
STUDENT_API_BASE=https://api.deepseek.com
STUDENT_MODEL=deepseek-chat
[/dim]""", title="Config Guide", border_style="dim"))
    
    console.print("""
AI will play a student answering:
  - Will make some common mistakes (careless calculation, formula errors, etc.)
  - Generated answers will be saved to file
""")
    
    if not Confirm.ask("[magenta]Simulate student answering?[/magenta]", default=False):
        return None
    
    use_custom_api = False
    if Confirm.ask("[magenta]Temporarily modify API config?[/magenta]", default=False):
        use_custom_api = True
        
        console.print("\n[bold]Enter new API config (press Enter to use current):[/bold]")
        
        new_base = Prompt.ask(
            "[magenta]API Base URL[/magenta]",
            default=config["api_base"] or ""
        ).strip()
        if new_base:
            config["api_base"] = new_base
        
        new_key = Prompt.ask(
            "[magenta]API Key[/magenta]",
            default="",
            password=True
        ).strip()
        if new_key:
            config["api_key"] = new_key
        
        new_model = Prompt.ask(
            "[magenta]Model Name[/magenta]",
            default=config["model"]
        ).strip()
        if new_model:
            config["model"] = new_model
        
        console.print(f"\n[green]API config updated[/green]")
        console.print(f"  Base: {config['api_base'] or 'Default'}")
        console.print(f"  Model: {config['model']}")
    
    if Confirm.ask("[magenta]Temporarily adjust accuracy rate?[/magenta]", default=False):
        rate_str = Prompt.ask(
            "[magenta]Enter accuracy rate[/magenta] (0-100 integer, e.g. 70)",
            default=str(config['correct_rate'])
        )
        try:
            config["correct_rate"] = int(rate_str)
            console.print(f"[green]Accuracy set to: {config['correct_rate']}%[/green]")
        except:
            console.print("[yellow]Invalid format, using default[/yellow]")
    
    if session_dir:
        default_path = os.path.join(session_dir, "simulated_student_answers.json")
    else:
        default_path = "simulated_student_answers.json"
    
    output_path = Prompt.ask(
        "[magenta]Answer file save path[/magenta]",
        default=default_path
    )
    
    try:
        if use_custom_api:
            student_client = create_student_llm_client(config)
            console.print(f"\n[dim]Using custom API: {config['api_base'] or 'OpenAI'} / {config['model']}[/dim]")
        else:
            student_client = llm_client
        
        answers = simulate_and_save_answers(
            llm_client=student_client,
            questions=questions,
            output_path=output_path,
            solve_results=solve_results,
            correct_rate=config["correct_rate"],
            subject=subject
        )
        return answers
    except Exception as e:
        console.print(f"[red]Simulation failed: {e}[/red]")
        return None
