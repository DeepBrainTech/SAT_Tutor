# SAT Tutor - Intelligent Problem Diagnosis System

An end-to-end problem processing system that supports PDF question extraction, automatic solving, and error diagnosis. Works with both **SAT Math** (vision-based) and **SAT English** (OCR-based).

## Features

- **PDF to Images**: Automatically convert PDF pages to high-resolution images
- **Question Extraction (Stage T)**: 
  - Math: Use GPT Vision to extract structured questions from images
  - English: Use OCR + Text LLM to extract questions from text
- **Intelligent Solving (Stage S)**: Automatically solve questions with key steps
- **Error Diagnosis (Stage D)**: Three powerful diagnosis modes
  - **Mode A**: Direct solution with complete steps
  - **Mode B**: Contrastive analysis of errors vs correct approach (default)
  - **Mode C**: Scaffolded tutoring with hints and cognitive anchors
- **Student Simulation**: Let AI simulate a student answering questions with realistic mistakes

## System Requirements

- Python 3.9+
- Poppler (for PDF to image conversion)
- Tesseract OCR (for SAT English mode only)

## Installation

### 1. Install Poppler (Required)

#### macOS
```bash
brew install poppler
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

#### Windows
1. Download Poppler for Windows: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\Program Files\poppler`
3. Add `C:\Program Files\poppler\Library\bin` to system PATH

### 2. Install Tesseract OCR (Optional - for SAT English)

#### macOS
```bash
brew install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr
```

#### Windows
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example config file and fill in your API Key:

```bash
cp .env.example .env
```

Edit `.env` file:

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL_VISION=gpt-4o
OPENAI_MODEL_TEXT=gpt-4o-mini
```

## Quick Start (Most Common Commands)

### SAT Math - Full Diagnosis
```bash
python -m gre_tutor.run --pdf data/samples/Linear_Equations.pdf --subject math --mode diagnose
```
(Replace `data/samples/Linear_Equations.pdf` with your own PDF file path)

### SAT English - Full Diagnosis
```bash
python -m gre_tutor.run --pdf data/samples/Cross.pdf --subject english --mode diagnose
```
(Replace `data/samples/Cross.pdf` with your own PDF file path)

### With Preset Correct Answers (Skip LLM Solving)
```bash
# Math
python -m gre_tutor.run --pdf data/samples/Linear_Equations.pdf --subject math --correct-answers data/samples/correct_answers_simple.json --mode diagnose

# English
python -m gre_tutor.run --pdf data/samples/Cross.pdf --subject english --correct-answers data/samples/correct_answers_simple.json --mode diagnose
```
(Replace file paths with your own PDF and answer JSON files)

After running, the system will:
1. Convert PDF to images
2. Extract questions (Vision for Math, OCR for English)
3. Solve questions (or use your correct answers file)
4. Ask you to input answers (interactive, file, or simulate student)
5. **Select diagnosis mode** (A/B/C - see below)
6. Diagnose errors and generate report

## Diagnosis Modes

The system offers **three diagnosis modes** to suit different learning needs. You can select your preferred mode during the diagnosis stage (after answering questions).

### Mode A: Direct Solution
**Best for:** Quick review, time-limited sessions, getting the answer immediately

**What you get:**
- ✓ Correct answer revealed directly
- ✓ Complete step-by-step solution
- ✓ One-sentence summary of key insight

**Example output:**
```
Correct Answer: C

Key Steps:
1. Identify the given ratio x:y = 3:2
2. Set up equation: x = 1.5y
3. Substitute into total constraint
4. Solve for y, then calculate x

Summary: This is a proportion problem - understanding the ratio relationship is key.
```

---

### Mode B: Contrastive Explanation - **Default**
**Best for:** Understanding common mistakes, learning from errors

**What you get:**
- ✓ Analysis of why your wrong answer seems tempting
- ✓ Explanation of what went wrong
- ✓ Comparison with the correct solution
- ✓ Option-by-option analysis (for multiple choice)

**Example output:**
```
Your Answer: B | Correct Answer: C

Why B is Tempting:
Option B appears correct if you only consider the first paragraph...

Likely Misconceptions:
1. Confusing correlation with causation
2. Overlooking the key qualifier in paragraph 2

How to Get Correct:
First, note that the author states... Then, compare this with...
```

---

### Mode C: Scaffolded Tutoring - **Interactive Learning**
**Best for:** Deep learning, building problem-solving skills, understanding concepts

**What you get - Two-stage process:**

**Stage 1: Hints (without revealing answer)**
- ✗ Correct answer is NOT revealed
- ✓ Error analysis: what went wrong
- ✓ **Actionable hints with cognitive anchors:**
  - Specific action to take
  - Where to find evidence
  - Guiding question
  - **Expected conclusion** (what you should understand)
- ✓ Second chance to answer

**Example hint output:**
```
Your first answer: B (incorrect)

Error Analysis:
You may have missed the relationship between forced voting and true preferences.

Next Steps to Try:

  Step 1: Re-read where Singh and Roy discuss forced voting
    Where to look: Text 2, paragraph discussing their research findings
    Think: What did Singh and Roy find about people who feel forced to vote?
    What you should understand: Based on their findings, decide whether 
    mandatory voting strengthens or weakens the link between votes and 
    voters' true preferences. You should understand the RELATIONSHIP 
    between forced voting and genuine preference expression.

  Step 2: Compare this finding with Fowler's argument
    Where to look: Text 1, where Fowler discusses election results
    Think: Do Singh and Roy agree or disagree with Fowler?
    What you should understand: You should identify the CONTRASTING 
    VIEWPOINT - one author sees a benefit, the other sees a drawback.

Try again with these hints!
```

**Stage 2: Complete Analysis (after second attempt)**
- ✓ Full solution with both attempts analyzed
- ✓ Explanation of first mistake
- ✓ Explanation of second attempt (if still wrong)
- ✓ Encouragement based on improvement

**Example final output:**
```
First Attempt (wrong): B
Second Attempt: C (correct!)

First attempt (B): You focused on the number of voters but missed the 
quality of their decision-making. Singh and Roy's key finding is about...

Second attempt (C): Correct! You successfully identified that mandatory 
voting increases quantity but may decrease quality of voter decisions.

Great improvement! You learned to look beyond surface-level information 
and consider the deeper implications of research findings.
```

---

### How to Select Mode

When you run the diagnosis command:

```bash
python -m gre_tutor.run --pdf input.pdf --mode diagnose
```

After entering your answers, you'll see:

```
════════════════════════════════════════════════════
Diagnosis Mode Selection
════════════════════════════════════════════════════

Condition A: Direct Solution
  - Show correct answer directly
  - Complete step-by-step solution
  - One-sentence summary
  Best for: Quick review, time-limited sessions

Condition B: Contrastive Explanation (Current Default)
  - Analyze why the wrong answer seems tempting
  - Explain what went wrong
  - Show correct solution with comparison
  Best for: Understanding common mistakes

Condition C: Scaffolded Tutoring
  - First give hints (without revealing answer)
  - Student gets a second chance to answer
  - Then reveal full solution and analysis
  Best for: Deep learning, building problem-solving skills

Please choose diagnosis mode [A/B/C]: _
```

**Key Features of Mode C (Scaffolded Tutoring):**
- **Cognitive Anchors**: Each hint includes an "expected conclusion" to help you understand the concept, not just follow steps
- **For Math**: Learn *why* the formula works, not just how to apply it
- **For English**: Understand the *relationship* between ideas, not just locate text
- **Tracks Progress**: Even if you get it right on the second try, the system remembers you were wrong initially for learning analytics

## Project Structure

```
gre-math-tutor/
├── gre_tutor/
│   ├── __init__.py
│   ├── run.py                    # CLI entry point
│   │
│   ├── core/                     # Core business logic
│   │   ├── models.py             # Pydantic data models (Question, SolveResult, DiagnoseResult)
│   │   ├── pipeline.py           # Main processing pipeline
│   │   ├── solver.py             # Stage S: Question solving logic
│   │   ├── diagnose.py           # Stage D: Error diagnosis logic
│   │   ├── validators.py         # JSON schema validation
│   │   └── taxonomy.py           # Error classification taxonomy
│   │
│   ├── ingest/                   # Input processing
│   │   ├── pdf_to_images.py      # PDF to PNG conversion
│   │   ├── page_range.py         # Page range parsing (e.g., "1-3,5")
│   │   ├── vision_extract.py     # Stage T: Vision-based extraction (Math)
│   │   ├── ocr_extract.py        # OCR text extraction (English)
│   │   └── text_extract.py       # Stage T: Text-based extraction (English)
│   │
│   ├── llm/                      # LLM integration
│   │   ├── base.py               # Abstract LLM client interface
│   │   ├── openai_client.py      # OpenAI API client (supports compatible APIs)
│   │   ├── mock_client.py        # Mock client for testing without API
│   │   └── prompts.py            # All prompt templates for each stage
│   │
│   ├── io/                       # Input/Output handling
│   │   ├── answers.py            # User answer collection (interactive/file)
│   │   ├── json_io.py            # JSON file operations
│   │   ├── report_md.py          # Markdown report generation
│   │   └── student_simulator.py  # AI student simulation
│   │
│   └── utils/                    # Utilities
│       ├── logging.py            # Logging utilities
│       └── time.py               # Timestamp generation
│
├── data/samples/                 # Sample data files
├── outputs/                      # Output directory
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
└── README.md                     # This file
```

## File Descriptions

### Core Modules (`gre_tutor/core/`)

| File | Description |
|------|-------------|
| `models.py` | Pydantic models: `Question`, `SolveResult`, `DiagnoseResult`, `SessionResult` |
| `pipeline.py` | Main pipeline orchestrating all stages (PDF→Extract→Solve→Diagnose) |
| `solver.py` | Question solving using LLM with step-by-step reasoning |
| `diagnose.py` | Error diagnosis comparing user answer vs correct answer |
| `validators.py` | JSON schema validation for LLM outputs |
| `taxonomy.py` | Classification of error types and misconceptions |

### Input Processing (`gre_tutor/ingest/`)

| File | Description |
|------|-------------|
| `pdf_to_images.py` | Convert PDF pages to PNG images using pdf2image |
| `page_range.py` | Parse page range strings like "1-3,5,7-10" |
| `vision_extract.py` | Extract Math questions using GPT Vision |
| `ocr_extract.py` | Extract text from images using Tesseract OCR |
| `text_extract.py` | Extract English questions from OCR text using LLM |

### LLM Integration (`gre_tutor/llm/`)

| File | Description |
|------|-------------|
| `base.py` | Abstract `LLMClient` interface |
| `openai_client.py` | OpenAI API implementation (supports DeepSeek, etc.) |
| `mock_client.py` | Mock responses for testing without API |
| `prompts.py` | All prompt templates for transcription, solving, diagnosis |

### I/O Handling (`gre_tutor/io/`)

| File | Description |
|------|-------------|
| `answers.py` | Collect user answers (interactive input or JSON file) |
| `json_io.py` | Save/load JSON files and session results |
| `report_md.py` | Generate human-readable Markdown reports |
| `student_simulator.py` | Simulate student answering with configurable accuracy |

## Usage Commands

### Basic Commands

```bash
# Full pipeline: Extract + Solve + Diagnose (interactive answer input)
python -m gre_tutor.run --pdf input.pdf --mode diagnose

# Transcribe only (extract questions)
python -m gre_tutor.run --pdf input.pdf --mode transcribe_only

# Transcribe + Solve (no diagnosis)
python -m gre_tutor.run --pdf input.pdf --mode solve
```

### Subject Selection

```bash
# GRE/SAT Math (default) - uses Vision LLM
python -m gre_tutor.run --pdf math_test.pdf --subject math

# SAT English - uses OCR + Text LLM
python -m gre_tutor.run --pdf english_test.pdf --subject english
```

### Page Selection

```bash
# Process specific pages
python -m gre_tutor.run --pdf input.pdf --pages "1-3,5,7"

# Process all pages (default)
python -m gre_tutor.run --pdf input.pdf --pages all
```

### Answer Input Methods

```bash
# Interactive input (default)
python -m gre_tutor.run --pdf input.pdf --mode diagnose

# Use preset user answers file
python -m gre_tutor.run --pdf input.pdf --answers user_answers.json --mode diagnose

# Use correct answers file (skip LLM solving)
python -m gre_tutor.run --pdf input.pdf --correct-answers correct.json --mode diagnose

# Both preset files (fully automated)
python -m gre_tutor.run --pdf input.pdf --correct-answers correct.json --answers user.json --no-interactive
```

### Advanced Options

```bash
# Higher image resolution (better OCR)
python -m gre_tutor.run --pdf input.pdf --dpi 400

# Custom output directory
python -m gre_tutor.run --pdf input.pdf --outdir my_outputs/

# Offline mode (mock data, no API needed)
python -m gre_tutor.run --pdf input.pdf --no-llm --mode transcribe_only

# Non-interactive mode (use all CLI arguments)
python -m gre_tutor.run --pdf input.pdf --no-interactive --correct-answers correct.json
```

### Student Simulation

During interactive mode, you can choose to simulate a student:

```bash
python -m gre_tutor.run --pdf input.pdf --mode diagnose
```

Then select option `[3] Simulate student` when prompted for answer input method.

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pdf` | PDF file path | **Required** |
| `--subject` | Subject type: `math` or `english` | `math` |
| `--mode` | Run mode: `transcribe_only`, `solve`, `diagnose` | `diagnose` |
| `--pages` | Page range, e.g., "1-3,5" or "all" | `all` |
| `--dpi` | Image resolution | `300` |
| `--outdir` | Output directory | `outputs/` |
| `--answers` | Preset user answers JSON file | None |
| `--correct-answers` | Preset correct answers JSON file | None |
| `--no-llm` | Force mock mode (no API needed) | `False` |
| `--no-interactive` | Disable interactive prompts | `False` |

## Output Structure

Each run creates a timestamped session folder:

```
outputs/
  session_20241216_143052/
    pages/                 # PDF converted images
      page_001.png
      page_002.png
    ocr_text.txt           # OCR extracted text (English mode only)
    transcribed.json       # Extracted questions
    results.json           # Solve + Diagnose results
    report.md              # Human-readable report
    logs.txt               # Run logs
    simulated_student_answers.json         # (if simulation used)
    simulated_student_answers_details.json # (detailed simulation output)
```

### Report Contents

The `report.md` file includes different sections based on the diagnosis mode used:

**All Modes:**
- Summary statistics (answered questions, correct count, accuracy)
- Question details with stems and options
- Solution steps

**Mode A (Direct Solution):**
- Correct answer
- Complete solution steps
- One-sentence summary

**Mode B (Contrastive Explanation) - Default:**
- Why the wrong answer is tempting
- Likely misconceptions
- How to get the correct answer
- Option-by-option analysis

**Mode C (Scaffolded Tutoring):**
- **First attempt** (marked as wrong even if second was correct)
- Second attempt
- Note if student improved
- Analysis of both attempts
- Complete solution with encouragement

**Mode C Statistics Section:**
```markdown
### Scaffolded Tutoring (Mode C) Statistics
- **First Attempt Wrong**: 5 questions
- **Questions**: p1_q1, p1_q3, p2_q2, p2_q4, p3_q1
- **Recovered on 2nd Attempt**: 3 questions
```

This helps track learning progress - even if a student gets it right on the second try, the initial mistake is recorded for analytics.

## Environment Variables

### Main API Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key | **Required** (unless `--no-llm`) |
| `OPENAI_API_BASE` | API base URL (for compatible APIs) | OpenAI default |
| `OPENAI_MODEL_VISION` | Vision model (for question extraction) | `gpt-4o` |
| `OPENAI_MODEL_TEXT` | Text model (for solving/diagnosis) | `gpt-4o-mini` |

### Student Simulation Configuration (Optional)

| Variable | Description | Default |
|----------|-------------|---------|
| `STUDENT_API_KEY` | Student simulation API key | Same as `OPENAI_API_KEY` |
| `STUDENT_API_BASE` | Student simulation API base URL | Same as `OPENAI_API_BASE` |
| `STUDENT_MODEL` | Student simulation model | Same as `OPENAI_MODEL_TEXT` |
| `STUDENT_CORRECT_RATE` | Target accuracy rate (0-100) | `70` |

### Example .env Configurations

#### OpenAI
```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL_VISION=gpt-4o
OPENAI_MODEL_TEXT=gpt-4o-mini
```

#### DeepSeek
```env
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL_VISION=deepseek-chat
OPENAI_MODEL_TEXT=deepseek-chat
```

#### Mixed (OpenAI for main, DeepSeek for student simulation)
```env
# Main API uses OpenAI
OPENAI_API_KEY=sk-openai-key
OPENAI_MODEL_VISION=gpt-4o
OPENAI_MODEL_TEXT=gpt-4o-mini

# Student simulation uses DeepSeek (cheaper)
STUDENT_API_KEY=sk-deepseek-key
STUDENT_API_BASE=https://api.deepseek.com
STUDENT_MODEL=deepseek-chat
STUDENT_CORRECT_RATE=70
```

### Supported API Platforms

| Platform | API Base URL | Example Models |
|----------|-------------|----------------|
| OpenAI | (default, no need to set) | gpt-4o, gpt-4o-mini |
| DeepSeek | https://api.deepseek.com | deepseek-chat |
| Zhipu AI | https://open.bigmodel.cn/api/paas/v4 | glm-4-flash |
| Moonshot | https://api.moonshot.cn/v1 | moonshot-v1-8k |
| Local Ollama | http://localhost:11434/v1 | llama3, qwen2 |

## Answer File Formats

### Simple Format (user answers or correct answers)
```json
{
  "p1_q1": "A",
  "p1_q2": "B",
  "p1_q3": "14",
  "p2_q1": "C"
}
```

### Detailed Format (correct answers with explanations)
```json
{
  "p1_q1": {
    "answer": "A",
    "topic": "algebra",
    "steps": ["Step 1...", "Step 2..."],
    "reason": "Because..."
  }
}
```

## Tips for Choosing Diagnosis Mode

### When to Use Mode A (Direct Solution)
✓ You need to check answers quickly  
✓ You're doing a timed practice session  
✓ You already understand the concepts but made a careless mistake  
✓ You want to see the solution method immediately  

### When to Use Mode B (Contrastive Explanation) - Default
✓ You want to understand why you made the mistake  
✓ You're learning from patterns of errors  
✓ You need detailed comparison between options  
✓ You're reviewing after a practice test  

### When to Use Mode C (Scaffolded Tutoring)
✓ You're learning a new concept  
✓ You have time for deep practice  
✓ You want to build problem-solving skills  
✓ You want to **understand the "why" behind the answer**, not just the steps  
✓ You prefer guided discovery over direct instruction  

