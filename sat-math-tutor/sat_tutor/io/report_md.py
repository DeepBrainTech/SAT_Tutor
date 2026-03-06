"""
Markdown Report Generation Module
Generate human-readable diagnosis reports
"""

import os
from datetime import datetime
from typing import Optional

from ..core.models import SessionResult, Question, SolveResult, DiagnoseResult


def generate_report_md(result: SessionResult) -> str:
    """
    Generate Markdown format diagnosis report
    
    Args:
        result: SessionResult object
    
    Returns:
        Markdown format report string
    """
    lines = []
    
    # Title
    lines.append("# SAT Tutor Diagnosis Report")
    lines.append("")
    
    # Basic info
    lines.append(f"**Generated Time**: {result.timestamp}")
    lines.append(f"**PDF File**: {os.path.basename(result.pdf_path)}")
    lines.append(f"**Run Mode**: {result.mode}")
    lines.append(f"**Processed Pages**: {result.transcribed.total_pages}")
    lines.append(f"**Total Questions**: {result.total_questions}")
    lines.append("")
    
    # Failed pages
    if result.transcribed.failed_pages:
        lines.append(f"**Warning: Failed Pages**: {', '.join(map(str, result.transcribed.failed_pages))}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Summary statistics (if has diagnosis results)
    if result.diagnose_results:
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Answered Questions**: {result.answered_questions}")
        lines.append(f"- **Correct Count**: {result.correct_count}")
        
        if result.answered_questions > 0:
            accuracy = result.correct_count / result.answered_questions * 100
            lines.append(f"- **Accuracy**: {accuracy:.1f}%")
        
        if result.incorrect_ids:
            lines.append(f"- **Wrong Questions**: {', '.join(result.incorrect_ids)}")
        
        # Mode C scaffolded tutoring statistics
        if result.first_attempt_wrong_count > 0:
            lines.append("")
            lines.append("### Scaffolded Tutoring (Mode C) Statistics")
            lines.append(f"- **First Attempt Wrong**: {result.first_attempt_wrong_count} questions")
            lines.append(f"- **Questions**: {', '.join(result.first_attempt_wrong_ids)}")
            # Calculate how many got it right on second attempt
            recovered = sum(1 for dr in result.diagnose_results 
                          if dr.first_attempt_wrong and dr.is_correct)
            if recovered > 0:
                lines.append(f"- **Recovered on 2nd Attempt**: {recovered} questions")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Question details
    lines.append("## Question Details")
    lines.append("")
    
    # Build mappings
    solve_map = {sr.question_id: sr for sr in result.solve_results}
    diagnose_map = {dr.question_id: dr for dr in result.diagnose_results}
    
    for question in result.transcribed.questions:
        q_id = question.id
        solve = solve_map.get(q_id)
        diagnose = diagnose_map.get(q_id)
        
        # Question title
        if diagnose:
            status = "CORRECT" if diagnose.is_correct else "WRONG"
        else:
            status = ""
        
        lines.append(f"### Question {q_id} {status}")
        lines.append("")
        
        # Stem
        lines.append(f"**Stem**: {question.stem}")
        lines.append("")
        
        # Options
        lines.append("**Options**:")
        correct_answer = solve.correct_answer if solve else None
        user_answer = diagnose.user_answer if diagnose else None
        
        for opt in ['A', 'B', 'C', 'D', 'E']:
            content = question.choices.get(opt, "")
            if content:
                markers = []
                if correct_answer and opt == correct_answer.upper():
                    markers.append("(correct)")
                if user_answer and opt == user_answer.upper() and opt != (correct_answer or "").upper():
                    markers.append("(user choice)")
                
                marker_str = " " + " ".join(markers) if markers else ""
                lines.append(f"- {opt}: {content}{marker_str}")
        
        lines.append("")
        
        # Answer info
        if diagnose:
            # Check if this was a Mode C scaffolded tutoring session
            if diagnose.first_attempt and diagnose.first_attempt_wrong:
                lines.append(f"**First Attempt (wrong)**: {diagnose.first_attempt}")
                lines.append(f"**Final Attempt**: {diagnose.user_answer} | **Correct Answer**: {diagnose.correct_answer}")
                if diagnose.is_correct:
                    lines.append("*Note: Student got it right after guided retries with hints*")
            else:
                lines.append(f"**User Answer**: {diagnose.user_answer} | **Correct Answer**: {diagnose.correct_answer}")
            lines.append("")
        elif solve:
            lines.append(f"**Correct Answer**: {solve.correct_answer}")
            lines.append("")
        
        if diagnose and (diagnose.student_work_image_path or diagnose.student_work_transcription):
            lines.append("**Student Handwritten Work (Uploaded)**:")
            if diagnose.student_work_image_path:
                lines.append(f"- Image: {diagnose.student_work_image_path}")
            if diagnose.student_work_transcription:
                lines.append("- Transcription:")
                lines.append(diagnose.student_work_transcription)
            lines.append("")
        
        # Solution steps
        if solve:
            lines.append("**Key Steps**:")
            for i, step in enumerate(solve.key_steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")
            lines.append(f"**Topic**: {solve.topic}")
            lines.append("")
        
        # Error diagnosis (only when wrong)
        if diagnose and not diagnose.is_correct:
            lines.append("#### Error Analysis")
            lines.append("")
            
            if diagnose.why_user_choice_is_tempting:
                lines.append(f"**Why {diagnose.user_answer} is Tempting**:")
                lines.append(diagnose.why_user_choice_is_tempting)
                lines.append("")
            
            if diagnose.likely_misconceptions:
                lines.append("**Likely Misconceptions**:")
                for i, misconception in enumerate(diagnose.likely_misconceptions, 1):
                    lines.append(f"{i}. {misconception}")
                lines.append("")
            
            if diagnose.how_to_get_correct:
                lines.append("**How to Get Correct Answer**:")
                lines.append(diagnose.how_to_get_correct)
                lines.append("")
            
            # Option analysis
            if diagnose.option_analysis:
                lines.append("**Option Analysis**:")
                for oa in diagnose.option_analysis:
                    status_mark = "(correct)" if oa.is_correct else "(user choice)" if oa.is_user_choice else ""
                    lines.append(f"- **{oa.option}**: {oa.analysis} {status_mark}")
                lines.append("")
        
        # Uncertain spans
        if question.uncertain_spans:
            lines.append("**Warning: Uncertain Spans**:")
            for span in question.uncertain_spans:
                lines.append(f"- \"{span.span}\" ({span.reason}) @ {span.location}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Error log
    if result.transcribed.errors:
        lines.append("## Error Log")
        lines.append("")
        for error in result.transcribed.errors:
            lines.append(f"- {error[:200]}...")
        lines.append("")
    
    return "\n".join(lines)


def save_report_md(result: SessionResult, file_path: str) -> None:
    """
    Save Markdown report to file
    
    Args:
        result: SessionResult object
        file_path: Output path
    """
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    
    report = generate_report_md(result)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report)


def print_summary(result: SessionResult) -> None:
    """
    Print run summary to console
    
    Args:
        result: SessionResult object
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    
    # Title
    console.print("\n" + "="*60, style="bold blue")
    console.print("SAT Tutor Run Summary", style="bold blue")
    console.print("="*60, style="bold blue")
    
    # Basic info table
    table = Table(show_header=False, box=None)
    table.add_column("Item", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("PDF File", os.path.basename(result.pdf_path))
    table.add_row("Run Mode", result.mode)
    table.add_row("Parsed Pages", str(result.transcribed.total_pages))
    table.add_row("Extracted Questions", str(result.total_questions))
    
    if result.transcribed.failed_pages:
        table.add_row("Failed Pages", ", ".join(map(str, result.transcribed.failed_pages)))
    
    console.print(table)
    console.print("")
    
    # Diagnosis statistics
    if result.diagnose_results:
        console.print("[bold]Answer Statistics:[/bold]")
        console.print(f"  Answered: {result.answered_questions}")
        console.print(f"  Correct: [green]{result.correct_count}[/green]")
        
        incorrect_count = result.answered_questions - result.correct_count
        if incorrect_count > 0:
            console.print(f"  Wrong: [red]{incorrect_count}[/red]")
            console.print(f"  Wrong IDs: [red]{', '.join(result.incorrect_ids)}[/red]")
        
        if result.answered_questions > 0:
            accuracy = result.correct_count / result.answered_questions * 100
            color = "green" if accuracy >= 80 else "yellow" if accuracy >= 60 else "red"
            console.print(f"  Accuracy: [{color}]{accuracy:.1f}%[/{color}]")
        
        console.print("")
    
    # Output path
    console.print("[bold]Output Directory:[/bold]")
    session_dir = f"outputs/session_{result.session_id}"
    console.print(f"  [dim]{session_dir}/[/dim]")
    console.print("")
    
    console.print("="*60, style="bold blue")
