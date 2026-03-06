"""
SAT Tutor - CLI Entry
"""

import argparse
import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from rich.panel import Panel

from .core.pipeline import GREMathPipeline


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="SAT Tutor - Intelligent Problem Diagnosis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SAT Math (default, uses vision model)
  python -m gre_tutor.run --pdf input.pdf --mode diagnose
  
  # SAT English (uses OCR + text model)
  python -m gre_tutor.run --pdf input.pdf --subject english --mode diagnose
  
  # Transcribe only
  python -m gre_tutor.run --pdf input.pdf --mode transcribe_only
  
  # Transcribe + Solve
  python -m gre_tutor.run --pdf input.pdf --mode solve
  
  # Preset user answers file (skip answer input selection)
  python -m gre_tutor.run --pdf input.pdf --answers user.json --mode diagnose
  
  # Preset correct answers file (skip solve selection)
  python -m gre_tutor.run --pdf input.pdf --correct-answers correct.json --mode diagnose
  
  # Non-interactive mode (fully use CLI arguments)
  python -m gre_tutor.run --pdf input.pdf --correct-answers correct.json --answers user.json --no-interactive
  
  # Specify page range
  python -m gre_tutor.run --pdf input.pdf --pages "1-3,5" --mode diagnose
  
  # Offline Mock mode
  python -m gre_tutor.run --pdf input.pdf --no-llm --mode transcribe_only
        """
    )
    
    # PDF argument (required unless --transcribed is provided)
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="PDF file path (optional if --transcribed is provided)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--subject",
        type=str,
        choices=["math", "english"],
        default="math",
        help="Subject type: math (SAT Math, vision extraction) or english (SAT English, OCR extraction)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["transcribe_only", "solve", "diagnose"],
        default="diagnose",
        help="Run mode: transcribe_only, solve (transcribe+solve), diagnose (full diagnosis, default)"
    )
    
    parser.add_argument(
        "--pages",
        type=str,
        default="all",
        help="Page range, e.g., '1-3,5' or 'all' (default: all)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution (default: 300)"
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs/)"
    )
    
    parser.add_argument(
        "--answers",
        type=str,
        default=None,
        help="Preset user answers JSON file path (skip answer input selection)"
    )
    
    parser.add_argument(
        "--correct-answers",
        type=str,
        default=None,
        dest="correct_answers",
        help="Preset correct answers JSON file path (skip correct answers selection, skip LLM solving)"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force Mock mode (no API Key needed)"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        dest="no_interactive",
        help="Disable interactive selection (fully use CLI arguments)"
    )
    
    parser.add_argument(
        "--transcribed",
        type=str,
        default=None,
        help="Load existing transcribed.json file (skip PDF conversion and extraction)"
    )
    
    return parser.parse_args()


def main():
    """Main entry function"""
    console = Console()
    
    # Show welcome message
    console.print(Panel.fit(
        "[bold blue]SAT Tutor[/bold blue]\n"
        "[dim]Intelligent Problem Diagnosis System[/dim]\n"
        "[dim]Supports: Math (Vision) | English (OCR)[/dim]",
        border_style="blue"
    ))
    console.print()
    
    # Parse arguments
    args = parse_args()
    
    # Validate: either PDF or transcribed must be provided
    if not args.pdf and not args.transcribed:
        console.print("[red]Error: Either --pdf or --transcribed must be provided[/red]")
        sys.exit(1)
    
    # Validate PDF file (if provided)
    if args.pdf and not os.path.exists(args.pdf):
        console.print(f"[red]Error: PDF file not found: {args.pdf}[/red]")
        sys.exit(1)
    
    # Validate transcribed file (if provided)
    if args.transcribed and not os.path.exists(args.transcribed):
        console.print(f"[red]Error: Transcribed file not found: {args.transcribed}[/red]")
        sys.exit(1)
    
    # Validate preset user answers file
    if args.answers and not os.path.exists(args.answers):
        console.print(f"[red]Error: User answers file not found: {args.answers}[/red]")
        sys.exit(1)
    
    # Validate preset correct answers file
    if args.correct_answers and not os.path.exists(args.correct_answers):
        console.print(f"[red]Error: Correct answers file not found: {args.correct_answers}[/red]")
        sys.exit(1)
    
    # Show configuration
    console.print("[bold]Run Configuration:[/bold]")
    if args.transcribed:
        console.print(f"  Transcribed File: {args.transcribed} [yellow](skip extraction)[/yellow]")
        if args.pdf:
            console.print(f"  PDF File: {args.pdf} [dim](not used)[/dim]")
    else:
        console.print(f"  PDF File: {args.pdf}")
    console.print(f"  Subject: {args.subject} ({'Vision LLM' if args.subject == 'math' else 'OCR + Text LLM'})")
    console.print(f"  Run Mode: {args.mode}")
    if not args.transcribed:
        console.print(f"  Page Range: {args.pages}")
        console.print(f"  Image DPI: {args.dpi}")
    console.print(f"  Output Dir: {args.outdir}")
    console.print(f"  Interactive: {'No' if args.no_interactive else 'Yes'}")
    if args.answers:
        console.print(f"  Preset User Answers: {args.answers}")
    if args.correct_answers:
        console.print(f"  Preset Correct Answers: {args.correct_answers}")
    console.print(f"  Mock Mode: {'Yes' if args.no_llm else 'No'}")
    console.print()
    
    # Check API Key (if LLM needed)
    need_llm_check = not args.no_llm and args.mode != "transcribe_only"
    if need_llm_check and not args.correct_answers:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-your"):
            console.print("[yellow]Warning: OPENAI_API_KEY not configured or invalid[/yellow]")
            console.print("[dim]Tip: Copy .env.example to .env and fill in your API Key[/dim]")
            console.print("[dim]Alternatively, provide a correct answers file in interactive mode[/dim]")
            console.print()
    
    try:
        # Create and run pipeline
        pipeline = GREMathPipeline(
            use_mock=args.no_llm,
            output_dir=args.outdir,
            subject=args.subject
        )
        
        result = pipeline.run(
            pdf_path=args.pdf or "",
            mode=args.mode,
            pages=args.pages,
            dpi=args.dpi,
            answers_json=args.answers,
            correct_answers_json=args.correct_answers,
            interactive=not args.no_interactive,
            transcribed_json=args.transcribed
        )
        
        console.print("\n[green]Processing complete![/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]User interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()