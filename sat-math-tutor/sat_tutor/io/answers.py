"""
User Answer Input Module
Supports interactive input and JSON file import
Supports multiple choice (A-E) and numeric entry (number/fraction/expression)
"""

import json
import os
import textwrap
from typing import Optional, Any

from ..core.models import Question
from ..core.validators import extract_json_from_text
from ..llm.prompts import (
    HANDWRITTEN_MATH_WORK_SYSTEM_PROMPT,
    HANDWRITTEN_MATH_WORK_USER_PROMPT_TEMPLATE,
    HANDWRITTEN_MATH_WORK_SCHEMA_HINT,
)


def load_answers_from_json(json_path: str) -> dict[str, str]:
    """
    Load answers from JSON file
    
    Args:
        json_path: JSON file path
    
    Returns:
        Answer dict {question_id: answer}
    
    Raises:
        FileNotFoundError: File does not exist
        json.JSONDecodeError: JSON parse failed
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Answer file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError("Answer file format error: expected dict format {question_id: answer}")
    
    answers = {}
    for q_id, answer in data.items():
        if answer is not None and not q_id.startswith("_"):
            answers[q_id] = str(answer).strip()
    
    return answers


def save_answers_to_json(answers: dict[str, str], json_path: str) -> None:
    """
    Save answers to JSON file
    
    Args:
        answers: Answer dict
        json_path: Output path
    """
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)


def wrap_text(text: str, width: int = 70) -> str:
    """
    Wrap long text for display
    
    Args:
        text: Original text
        width: Max width per line
    
    Returns:
        Wrapped text
    """
    lines = text.split('\n')
    wrapped_lines = []
    for line in lines:
        if len(line) <= width:
            wrapped_lines.append(line)
        else:
            wrapped = textwrap.fill(line, width=width, break_long_words=False, break_on_hyphens=False)
            wrapped_lines.append(wrapped)
    return '\n'.join(wrapped_lines)


def display_all_questions(questions: list[Question]) -> None:
    """
    Display all questions (for preview before batch input)
    
    Args:
        questions: Question list
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    
    console = Console(width=100)
    
    console.print("\n" + "="*70, style="blue")
    console.print("All Questions Preview", style="bold blue")
    console.print("="*70, style="blue")
    
    for i, question in enumerate(questions, 1):
        problem_type = question.problem_type
        is_numeric = problem_type == "numeric_entry"
        type_label = "[Numeric]" if is_numeric else "[Choice]"
        type_color = "magenta" if is_numeric else "cyan"
        
        console.print(f"\n[bold {type_color}]{i}. {question.id}[/bold {type_color}] {type_label}")
        
        stem_wrapped = wrap_text(question.stem, width=65)
        console.print(Panel(stem_wrapped, border_style="dim", padding=(0, 1)))
        
        if not is_numeric and question.choices:
            for opt in ['A', 'B', 'C', 'D', 'E']:
                content = question.choices.get(opt)
                if content and content not in ["N/A", "UNKNOWN", None]:
                    if len(content) > 60:
                        content = content[:57] + "..."
                    console.print(f"    [yellow]{opt}[/yellow]: {content}")
    
    console.print("\n" + "="*70, style="blue")
    
    console.print("\n[bold]Answer file format example:[/bold]")
    console.print('''
```json
{
  "p1_q1": "A",
  "p1_q2": "B", 
  "p1_q3": "7",
  "p1_q4": "1/2"
}
```
''', style="dim")


def ask_correct_answers_choice(questions: list[Question]) -> Optional[str]:
    """
    Ask user if they want to provide correct answers file
    
    Args:
        questions: Question list
    
    Returns:
        Correct answers file path, or None if user chooses not to provide
    """
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    
    console = Console()
    
    console.print("\n" + "="*70, style="yellow")
    console.print("Correct Answers Option", style="bold yellow")
    console.print("="*70, style="yellow")
    
    console.print("""
You can choose:
  [1] Provide correct answers file -> Skip LLM solving, use your answers directly
  [2] Don't provide -> Use LLM to solve questions automatically
""")
    
    choice = Prompt.ask(
        "[yellow]Please choose[/yellow]",
        choices=["1", "2"],
        default="2"
    )
    
    if choice == "1":
        console.print("\n[dim]Answer file format: JSON, e.g. {\"p1_q1\": \"A\", \"p1_q2\": \"7\"}[/dim]")
        
        while True:
            file_path = Prompt.ask(
                "[yellow]Enter correct answers file path[/yellow]",
                default=""
            )
            
            if not file_path:
                if Confirm.ask("[yellow]Sure you don't want to provide correct answers file?[/yellow]", default=True):
                    return None
                continue
            
            if not os.path.exists(file_path):
                console.print(f"[red]File not found: {file_path}[/red]")
                continue
            
            try:
                answers = load_answers_from_json(file_path)
                console.print(f"[green]Successfully loaded {len(answers)} correct answers[/green]")
                
                loaded_ids = set(answers.keys())
                question_ids = {q.id for q in questions}
                missing = question_ids - loaded_ids
                if missing:
                    console.print(f"[yellow]Warning: Missing correct answers for: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}[/yellow]")
                
                return file_path
            except Exception as e:
                console.print(f"[red]File format error: {e}[/red]")
                continue
    
    return None


def ask_user_answers_choice(
    questions: list[Question],
    llm_client=None,
    solve_results=None,
    session_dir: str = None,
    subject: str = "math"
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """
    Ask user to choose answer input method
    
    Args:
        questions: Question list
        llm_client: LLM client (for student simulation)
        solve_results: Correct answers list (for student simulation reference)
        session_dir: Session directory (for saving simulated answers)
        subject: "math" or "english" - determines which prompts to use for simulation
    
    Returns:
        (user answers dict, student handwritten work dict)
    """
    from rich.console import Console
    from rich.prompt import Prompt
    
    console = Console()
    
    console.print("\n" + "="*70, style="green")
    console.print("User Answer Input Method", style="bold green")
    console.print("="*70, style="green")
    
    options_text = """
You can choose:
  [1] Interactive input -> Display and input answers one by one
  [2] Batch file input  -> Display all questions first, then input answer file path"""
    
    has_simulator = llm_client is not None
    if has_simulator:
        options_text += """
  [3] Simulate student -> Let AI play student (will intentionally make some mistakes)"""
    
    console.print(options_text)
    console.print()
    
    valid_choices = ["1", "2", "3"] if has_simulator else ["1", "2"]
    choice = Prompt.ask(
        "[green]Please choose[/green]",
        choices=valid_choices,
        default="1"
    )
    
    if choice == "1":
        return collect_answers_interactive(
            questions,
            llm_client=llm_client,
            subject=subject
        )
    elif choice == "2":
        return collect_answers_from_file(questions), {}
    elif choice == "3" and has_simulator:
        from .student_simulator import ask_simulate_student
        simulated_answers = ask_simulate_student(
            llm_client=llm_client,
            questions=questions,
            solve_results=solve_results,
            session_dir=session_dir,
            subject=subject
        )
        if simulated_answers:
            return simulated_answers, {}
        else:
            console.print("[yellow]Falling back to interactive input...[/yellow]")
            return collect_answers_interactive(
                questions,
                llm_client=llm_client,
                subject=subject
            )
    else:
        return collect_answers_interactive(
            questions,
            llm_client=llm_client,
            subject=subject
        )


def collect_answers_from_file(questions: list[Question]) -> dict[str, str]:
    """
    Display all questions then load user answers from file
    
    Args:
        questions: Question list
    
    Returns:
        User answers dict
    """
    from rich.console import Console
    from rich.prompt import Prompt
    
    console = Console()
    
    display_all_questions(questions)
    
    console.print("\n[bold]Please prepare your answer file, then enter file path[/bold]")
    console.print("[dim]Tip: Answer file format is JSON, refer to question IDs above[/dim]\n")
    
    while True:
        file_path = Prompt.ask(
            "[green]Enter user answers file path[/green]",
            default=""
        )
        
        if not file_path:
            console.print("[yellow]No answer file provided, using empty answers[/yellow]")
            return {}
        
        if not os.path.exists(file_path):
            console.print(f"[red]File not found: {file_path}[/red]")
            continue
        
        try:
            answers = load_answers_from_json(file_path)
            console.print(f"\n[green]Successfully loaded {len(answers)} answers[/green]")
            
            console.print("\n[bold]Loaded answers:[/bold]")
            for q_id, ans in list(answers.items())[:10]:
                console.print(f"  {q_id}: {ans}")
            if len(answers) > 10:
                console.print(f"  ... total {len(answers)} answers")
            
            return answers
        except Exception as e:
            console.print(f"[red]Load failed: {e}[/red]")
            continue


def _transcribe_handwritten_work_image(
    llm_client,
    image_path: str,
    question_id: str
) -> dict[str, Any]:
    """
    Transcribe student's handwritten math work from an image using vision LLM.
    Returns a structured dict and never raises (stores errors in result).
    """
    result: dict[str, Any] = {
        "image_path": image_path,
        "transcribed_work": "",
        "step_lines": [],
        "unclear_parts": [],
        "confidence": 0.0,
        "error": None
    }
    
    if llm_client is None:
        result["error"] = "LLM client unavailable"
        return result
    
    if not os.path.exists(image_path):
        result["error"] = f"Image file not found: {image_path}"
        return result
    
    try:
        response = llm_client.generate_json(
            system_prompt=HANDWRITTEN_MATH_WORK_SYSTEM_PROMPT,
            user_prompt=HANDWRITTEN_MATH_WORK_USER_PROMPT_TEMPLATE.format(question_id=question_id),
            schema_hint=HANDWRITTEN_MATH_WORK_SCHEMA_HINT,
            images=[image_path],
            temperature=0.0
        )
        if not response.success:
            result["error"] = response.error or "Vision transcription failed"
            return result
        
        content = response.content or ""
        extracted = extract_json_from_text(content) or content
        data = json.loads(extracted)
        
        result["transcribed_work"] = str(data.get("transcribed_work", "")).strip()
        result["step_lines"] = [str(x) for x in data.get("step_lines", []) if str(x).strip()]
        result["unclear_parts"] = [str(x) for x in data.get("unclear_parts", []) if str(x).strip()]
        try:
            result["confidence"] = float(data.get("confidence", 0.0))
        except Exception:
            result["confidence"] = 0.0
        return result
    except Exception as e:
        result["error"] = f"Handwritten work transcription parse failed: {e}"
        return result


def collect_answers_interactive(
    questions: list[Question],
    llm_client=None,
    subject: str = "math"
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """
    Interactive collect user answers
    Supports multiple choice (A-E) and numeric entry (number/fraction/expression)
    
    Args:
        questions: Question list
    
    Returns:
        (answers dict, handwritten work dict)
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.markdown import Markdown
    
    console = Console(width=100)
    answers = {}
    handwritten_work = {}
    
    console.print("\n" + "="*70, style="blue")
    console.print("Interactive Answer Input", style="bold blue")
    console.print("   Multiple choice: Enter A-E", style="dim")
    console.print("   Numeric entry: Enter number, fraction (e.g. 1/2) or expression", style="dim")
    console.print("   Press Enter to skip, enter 'q' to quit", style="dim")
    console.print("="*70 + "\n", style="blue")
    
    for i, question in enumerate(questions, 1):
        problem_type = question.problem_type
        is_numeric = problem_type == "numeric_entry"
        
        type_label = "[Numeric Entry]" if is_numeric else "[Multiple Choice]"
        type_color = "magenta" if is_numeric else "cyan"
        
        console.print(f"\n{'-'*70}")
        console.print(f"[bold {type_color}]Question {i}/{len(questions)}[/bold {type_color}] {type_label} [dim]{question.id}[/dim]")
        console.print()
        
        stem_wrapped = wrap_text(question.stem, width=65)
        console.print(Panel(
            stem_wrapped, 
            title="[bold]Stem[/bold]", 
            border_style="dim",
            padding=(0, 1)
        ))
        
        if question.latex_equations:
            console.print("[dim]Formulas:[/dim]", end=" ")
            console.print(", ".join(question.latex_equations), style="italic")
        
        if question.diagram_description:
            console.print(f"[dim]Diagram:[/dim] {question.diagram_description}")
        
        if not is_numeric:
            choices = question.choices
            console.print()
            for opt in ['A', 'B', 'C', 'D', 'E']:
                content = choices.get(opt)
                if content and content not in ["N/A", "UNKNOWN", None]:
                    if len(content) > 60:
                        content_wrapped = wrap_text(content, width=55)
                        lines = content_wrapped.split('\n')
                        console.print(f"  [yellow]{opt}[/yellow]: {lines[0]}")
                        for line in lines[1:]:
                            console.print(f"      {line}")
                    else:
                        console.print(f"  [yellow]{opt}[/yellow]: {content}")
        
        console.print()
        
        while True:
            if is_numeric:
                prompt_text = "[green]Your answer (number/fraction)[/green]"
            else:
                prompt_text = "[green]Your answer (A-E)[/green]"
            
            answer = Prompt.ask(
                prompt_text,
                default="",
                show_default=False
            )
            
            answer = answer.strip()
            
            if answer.upper() == 'Q':
                console.print("\n[yellow]Exited answering[/yellow]")
                return answers, handwritten_work
            
            if answer == '':
                console.print("[dim]Skipped[/dim]")
                break
            
            if is_numeric:
                answers[question.id] = answer
                console.print(f"[green]Recorded: {answer}[/green]")
                break
            else:
                answer_upper = answer.upper()
                if answer_upper in ['A', 'B', 'C', 'D', 'E']:
                    answers[question.id] = answer_upper
                    console.print(f"[green]Recorded: {answer_upper}[/green]")
                    break
                else:
                    console.print("[red]Please enter one of A-E[/red]")

        # Optional handwritten work image upload (math interactive mode only)
        if (
            question.id in answers
            and subject == "math"
            and llm_client is not None
        ):
            console.print()
            upload_choice = Prompt.ask(
                "[cyan]Handwritten work image[/cyan] [dim](1=skip, 2=upload and transcribe)[/dim]",
                choices=["1", "2"],
                default="1"
            )
            if upload_choice == "2":
                image_path = Prompt.ask(
                    "[cyan]Enter handwritten work image path[/cyan]",
                    default="",
                    show_default=False
                ).strip()
                if image_path:
                    work_info = _transcribe_handwritten_work_image(
                        llm_client=llm_client,
                        image_path=image_path,
                        question_id=question.id
                    )
                    handwritten_work[question.id] = work_info
                    if work_info.get("error"):
                        console.print(f"[yellow]Handwritten work transcription failed:[/yellow] {work_info['error']}")
                    else:
                        preview = work_info.get("transcribed_work", "") or "No transcription returned"
                        if len(preview) > 180:
                            preview = preview[:177] + "..."
                        console.print("[green]Handwritten work transcribed and saved.[/green]")
                        console.print(f"[dim]{preview}[/dim]")
    
    console.print("\n" + "="*70, style="blue")
    console.print(f"Answering complete!", style="bold green")
    
    choice_count = sum(1 for q in questions if q.problem_type != "numeric_entry" and q.id in answers)
    numeric_count = sum(1 for q in questions if q.problem_type == "numeric_entry" and q.id in answers)
    
    console.print(f"   Total answered {len(answers)} questions (Multiple choice: {choice_count}, Numeric entry: {numeric_count})")
    console.print("="*70 + "\n", style="blue")
    
    return answers, handwritten_work


def merge_answers(
    existing: dict[str, str],
    new: dict[str, str],
    overwrite: bool = True
) -> dict[str, str]:
    """
    Merge answers
    
    Args:
        existing: Existing answers
        new: New answers
        overwrite: Whether to overwrite existing answers
    
    Returns:
        Merged answers
    """
    result = existing.copy()
    for q_id, answer in new.items():
        if overwrite or q_id not in result:
            result[q_id] = answer
    return result


# ============================================================
# Diagnosis Mode Selection
# ============================================================

DiagnoseMode = str  # "A", "B", or "C"


def ask_diagnose_mode() -> DiagnoseMode:
    """
    Ask user to choose diagnosis mode
    
    Returns:
        Diagnosis mode: "A", "B", or "C"
    """
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.panel import Panel
    
    console = Console()
    
    console.print("\n" + "="*70, style="magenta")
    console.print("Diagnosis Mode Selection", style="bold magenta")
    console.print("="*70, style="magenta")
    
    mode_descriptions = """
[bold cyan]Condition A: Direct Solution[/bold cyan]
  - Show correct answer directly
  - Complete step-by-step solution
  - One-sentence summary
  [dim]Best for: Quick review, time-limited sessions[/dim]

[bold yellow]Condition B: Contrastive Explanation (Current Default)[/bold yellow]
  - Analyze why the wrong answer seems tempting
  - Explain what went wrong
  - Show correct solution with comparison
  [dim]Best for: Understanding common mistakes[/dim]

[bold green]Condition C: Scaffolded Tutoring[/bold green]
  - First give hints (without revealing answer)
  - Student gets a second chance to answer
  - Then reveal full solution and analysis
  [dim]Best for: Deep learning, building problem-solving skills[/dim]
"""
    
    console.print(Panel(mode_descriptions, title="Choose a Mode", border_style="magenta"))
    
    choice = Prompt.ask(
        "[magenta]Please choose diagnosis mode[/magenta]",
        choices=["A", "B", "C", "a", "b", "c"],
        default="B"
    )
    
    return choice.upper()


def collect_second_attempt(
    question: Question,
    first_answer: str,
    hint_result: dict
) -> str:
    """
    For Condition C: Collect student's next attempt after showing hints.
    Can be called repeatedly in multi-round scaffolded tutoring.
    
    Args:
        question: The question
        first_answer: Student's previous answer
        hint_result: The hint/error analysis from LLM
    
    Returns:
        Student's next attempt answer
    """
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.panel import Panel
    
    console = Console(width=100)
    
    console.print("\n" + "="*70, style="yellow")
    console.print("Try Again With Hints", style="bold yellow")
    console.print("="*70, style="yellow")
    
    # Show the question again
    stem_wrapped = wrap_text(question.stem, width=65)
    console.print(Panel(stem_wrapped, title="[bold]Question[/bold]", border_style="cyan"))
    
    # Show options
    is_numeric = question.problem_type == "numeric_entry"
    if not is_numeric and question.choices:
        console.print()
        for opt in ['A', 'B', 'C', 'D', 'E']:
            content = question.choices.get(opt)
            if content and content not in ["N/A", "UNKNOWN", None]:
                console.print(f"  [yellow]{opt}[/yellow]: {content}")
    
    # Show error analysis
    console.print()
    console.print(Panel(
        f"[bold red]Your previous answer:[/bold red] {first_answer}\n\n" +
        f"[bold]Error Analysis:[/bold]\n{hint_result.get('error_analysis', 'N/A')}\n\n" +
        f"[bold]Key Concept:[/bold]\n{hint_result.get('key_concept_reminder', 'N/A')}",
        title="[bold yellow]Feedback[/bold yellow]",
        border_style="yellow"
    ))
    
    # Show actionable hints (new format with step/action/evidence/question/expected_conclusion)
    actionable_hints = hint_result.get('actionable_hints', [])
    if actionable_hints:
        console.print("\n[bold cyan]Next Steps to Try:[/bold cyan]")
        for hint in actionable_hints:
            step_num = hint.get('step_number', '?')
            action = hint.get('action', '')
            evidence = hint.get('evidence_location', '')
            guiding_q = hint.get('guiding_question', '')
            expected_conclusion = hint.get('expected_conclusion', '')
            
            console.print(f"\n  [bold]Step {step_num}:[/bold] {action}")
            if evidence:
                console.print(f"    [dim]Where to look:[/dim] {evidence}")
            if guiding_q:
                console.print(f"    [italic cyan]Think:[/italic cyan] {guiding_q}")
            if expected_conclusion:
                console.print(f"    [bold green]What you should understand:[/bold green] {expected_conclusion}")
    else:
        # Fallback to old hints format
        hints = hint_result.get('hints', [])
        if hints:
            console.print("\n[bold cyan]Hints:[/bold cyan]")
            for i, hint in enumerate(hints, 1):
                console.print(f"  {i}. {hint}")
    
    # Show encouragement
    try_again = hint_result.get('try_again_prompt', 'Take your time and try again!')
    console.print(f"\n[green]{try_again}[/green]\n")
    
    # Collect second attempt
    while True:
        if is_numeric:
            prompt_text = "[green]Your second answer (number/fraction)[/green]"
        else:
            prompt_text = "[green]Your second answer (A-E)[/green]"
        
        answer = Prompt.ask(prompt_text, default="", show_default=False)
        answer = answer.strip()
        
        if answer == '':
            console.print("[dim]Keeping previous answer[/dim]")
            return first_answer
        
        if is_numeric:
            return answer
        else:
            answer_upper = answer.upper()
            if answer_upper in ['A', 'B', 'C', 'D', 'E']:
                return answer_upper
            else:
                console.print("[red]Please enter one of A-E[/red]")
