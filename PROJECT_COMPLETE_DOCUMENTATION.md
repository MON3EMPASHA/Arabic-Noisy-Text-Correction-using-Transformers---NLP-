# Arabic Text Correction Transformer Project - Complete Documentation

**Date Created:** April 20, 2026  
**Project Location:** `e:\Programming\NLP\`  
**Status:** Production-Ready with Deployment

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation & Setup](#installation--setup)
4. [Core Components](#core-components)
5. [Technical Architecture](#technical-architecture)
6. [Data Pipeline](#data-pipeline)
7. [Model Training](#model-training)
8. [Deployment](#deployment)
9. [Build System](#build-system)
10. [File Descriptions](#file-descriptions)

---

## Project Overview

### Mission

This project implements an advanced Arabic text correction system using two complementary approaches:

1. **Custom Transformer Model** - A character-level seq2seq encoder-decoder architecture trained from scratch on synthetic Arabic noisy-clean pairs
2. **Pre-trained LLM Service** - CAMeL-Lab AraBART (CAMeL-Lab/arabart-qalb15-gec-ged-13) integrated for production-grade correction

### Applications

- Digital education platforms
- Media quality control
- Assistive language technology
- OCR post-processing
- Speech-to-text correction
- Real-time content moderation

### Key Features

- **Multiple Input Methods**: Manual text, file upload, image OCR, audio upload, live recording
- **Comprehensive Metrics**: CER, WER, BLEU, accuracy, precision, recall, F1-score
- **Error Analysis**: Character confusion matrix, error distribution visualization
- **Production UI**: Streamlit-based web application
- **Real-time Processing**: Multi-modal input handling

---

## Project Structure

```
e:\Programming\NLP\
├── app.py                                    # Streamlit application (Pre-trained LLM)
├── README.md                                 # Project overview and features
├── TECHNICAL_DOCUMENTATION.md                # Deep technical details
├── paper.md                                  # Paper abstract and methodology
├── requirements.txt                          # Python dependencies
├── packages.txt                              # System-level dependencies
├── NLP_Arabic_Text_Correction_Traditional.ipynb  # Custom Transformer training notebook
├── NLP_Project_LLM.ipynb                     # Pre-trained model experiments
├── .venv/                                    # Python virtual environment
├── .python-version                           # Python version specification
├── .git/                                     # Git repository
└── paper/                                    # Academic paper directory
    ├── paper.tex                             # IEEE conference paper (main)
    ├── paper.pdf                             # Compiled PDF output
    ├── paper.aux                             # LaTeX auxiliary file
    ├── paper.log                             # LaTeX build log
    ├── paper.fdb_latexmk                     # Latexmk database
    ├── paper.fls                             # File list for latexmk
    ├── combile.bat                           # Build script (Windows)
    ├── dl_paper.tex                          # Reference IEEE format template
    ├── NLP Paper hour1.docx                  # Original Word document
    └── Diagrams/                             # Figures and visualizations
        ├── Transfromer.jpg                   # Transformer architecture diagram
        ├── example.jpg                       # Encoder-decoder block diagram
        ├── CER.png                           # Character Error Rate distribution
        └── CM.png                            # Confusion matrix visualization
```

---

## Installation & Setup

### Prerequisites

- **Python:** 3.8 or higher
- **System Dependencies:**
  - FFmpeg (for audio processing)
  - LaTeX/MiKTeX (for paper compilation)
- **GPU:** Optional but recommended for training

### Environment Setup

```bash
# 1. Clone/Navigate to project
cd e:\Programming\NLP

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)

```
--extra-index-url https://download.pytorch.org/whl/cpu

streamlit
transformers==4.36.2
torch==2.7.1
jiwer
pillow
requests
openai-whisper
ffmpeg-python
streamlit-mic-recorder
pandas
plotly
soundfile
```

### System Dependencies (`packages.txt`)

```
ffmpeg
```

---

## Core Components

### 1. Streamlit Application (`app.py`)

**Purpose:** User-facing web interface for text correction  
**Framework:** Streamlit  
**Model:** CAMeL-Lab AraBART (pre-trained seq2seq)

**Key Functions:**

```python
def load_corrector():
    """Load pre-trained AraBART model and tokenizer (cached)"""
    model_name = "CAMeL-Lab/arabart-qalb15-gec-ged-13"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return corrector

def correct_text(text):
    """Correct Arabic text with chunk-based processing"""
    # Remove punctuation before correction
    # Process in 512-token chunks
    # Handle model outputs
    # Return corrected text

def ocr_space_image(image_bytes, api_key='helloworld', language='ara'):
    """Call OCR.space API for image text extraction"""

def calculate_detailed_metrics(reference, prediction):
    """Calculate word-level and character-level metrics"""
```

**Input Methods:**

1. Manual text input with examples
2. Text file upload (.txt)
3. Image OCR (OCR.space API)
4. Audio file upload (Whisper)
5. Live microphone recording

**Output Metrics:**

- Word accuracy
- Character accuracy
- Precision, Recall, F1-score
- Character count comparisons

---

### 2. Custom Transformer Model

**Notebook:** `NLP_Arabic_Text_Correction_Traditional.ipynb`

**Architecture:**

| Component              | Configuration  |
| ---------------------- | -------------- |
| Embedding Dimension    | 256            |
| Encoder Layers         | 3              |
| Decoder Layers         | 3              |
| Attention Heads        | 8              |
| Feed-Forward Dimension | 512            |
| Dropout                | 0.1            |
| Max Sequence Length    | 128            |
| Optimizer              | Adam (lr=1e-4) |
| Training Epochs        | 10             |
| Batch Size             | 32             |

**Key Features:**

1. **Text Normalization**
   - Alef variant normalization: إ, أ, آ → ا
   - Yaa variant normalization: ى → ي
   - Waw variant normalization: ؤ → و
   - Hamza normalization: ئ → ي
   - Taa Marbuta normalization: ة → ه
   - Diacritic removal (Fatha, Damma, Kasra, etc.)
   - Tatweel removal: ـ

2. **Character-Level Tokenization**
   - Vocabulary built from unique characters
   - Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
   - Vocabulary size: ~304 characters

3. **Training Strategy**
   - Scheduled sampling (teacher forcing decay: 1.0 → 0.3)
   - Mixed precision training (CUDA AMP)
   - Model checkpointing after each epoch
   - Resume from checkpoint capability

**Performance:**

```
Training Accuracy: 38.38% → 91.69% (peak at epoch 6) → 89.36% (epoch 10)
Character Error Rate (CER): 0.0364
BLEU Score: 0.292 (average)
Training Time: ~100 minutes on GPU
Model Size: ~50 MB
```

---

### 3. Pre-trained LLM Path

**Model:** CAMeL-Lab/arabart-qalb15-gec-ged-13  
**Source:** Hugging Face Hub  
**Base Architecture:** BART (Denoising Seq2Seq)

**Advantages:**

- Pre-trained on large Arabic corpora
- Fine-tuned on QALB datasets
- Strong contextual understanding
- Better grammatical fluency
- Production-ready

**Performance:**

```
Character Error Rate (CER): 0.0950
Model Size: ~1.5 GB
Training Time: 0 min (no fine-tuning needed)
Inference Speed: Moderate (GPU-dependent)
```

---

## Technical Architecture

### Transformer Encoder-Decoder

**Encoder:**

- Input: Noisy Arabic character sequence
- Output: Context-rich hidden representations
- Components:
  - Multi-head self-attention (8 heads, 32 dims each)
  - Position-wise feed-forward networks
  - Layer normalization and residual connections
  - Dropout regularization (10%)

**Decoder:**

- Input: Partially corrected output (auto-regressive)
- Components:
  - Masked self-attention (prevents lookahead)
  - Encoder-decoder cross-attention (source-target alignment)
  - Feed-forward networks
  - Output layer: Linear projection to vocabulary

**Why This Architecture?**

1. **Self-Attention:** Learns long-range character dependencies
2. **Cross-Attention:** Aligns generated output with noisy input context
3. **Masking:** Ensures causal generation (character-by-character)
4. **Residual Paths:** Preserves gradient flow in deep layers

### Data Pipeline

**Source:** Youm7.com (100,000 Arabic news articles)

**Processing Steps:**

1. **Collection**
   - Asynchronous web scraping (aiohttp + BeautifulSoup)
   - Polite crawling: 1-3s delays, robots.txt respect
   - Automatic retry for failed requests

2. **Cleaning**
   - Boilerplate removal (CSS selectors)
   - Unicode validation (Arabic character ranges)
   - Non-Arabic content filtering (>15% Latin threshold)
   - Space normalization

3. **Filtering**
   - Minimum 3 paragraphs per article
   - Minimum 50 words per article
   - Result: 100,000 high-quality articles

4. **Noise Injection**
   - **Character-level:** 7% error rate
     - Alef/Hamza confusions
     - Yaa/Alif confusions
     - Teh/Heh confusions
     - Visual and phonetic similarities
   - **Punctuation-level:** 15% modification rate
     - Spacing variations
     - Comma shape alternation (، vs ,)
     - Emphasis doubling
   - **Result:** 10,000 parallel noisy-clean pairs

**Error Types:**

| Error Category   | Example            |
| ---------------- | ------------------ |
| Alef Confusion   | جامعة → جامعه      |
| Hamza Placement  | تفاؤل → تفاءل      |
| Yaa/Alif         | قضى → قضا          |
| Whitespace       | ال نص → النص       |
| Morpho-syntactic | خمس كتب → خمسة كتب |
| Lexical/Semantic | ظفدع → ضفدع        |

---

## Model Training

### Custom Transformer Training Pipeline

**Notebook:** `NLP_Arabic_Text_Correction_Traditional.ipynb`

**Steps:**

1. **Data Loading**

   ```python
   train_data = 10,000 noisy-clean pairs
   - 80% training (8,000)
   - 10% validation (1,000)
   - 10% test (1,000)
   ```

2. **Preprocessing**

   ```python
   def preprocess(text):
       - Normalize Arabic characters
       - Encode to character indices
       - Pad to MAX_SEQ_LEN=128
   ```

3. **Training Loop**

   ```python
   for epoch in range(10):
       for batch in training_data:
           - Forward pass (encode → decode)
           - Compute loss (cross-entropy)
           - Backward pass
           - Update weights (Adam)
           - Scheduled sampling decay
           - Mixed precision scaling
   ```

4. **Evaluation**
   - Per-epoch accuracy tracking
   - Loss monitoring
   - Checkpoint saving (best model)
   - CER/BLEU calculation on test set

**Scheduled Sampling:**

```
Epoch 1: Teacher forcing ratio = 1.0 (always use reference)
Epoch 10: Teacher forcing ratio = 0.3 (use predictions 70% of the time)
```

**Mixed Precision Benefits:**

- 50% memory reduction
- 2x+ throughput improvement
- No significant accuracy loss (automatic scaling)

---

## Deployment

### Streamlit Application

**Launch Command:**

```bash
streamlit run app.py
```

**Default Port:** http://localhost:8501

**Tabs/Sections:**

1. **Manual Text Input**
   - Free-form Arabic text entry
   - Pre-defined example templates
   - Real-time correction

2. **File Upload**
   - .txt file input
   - Batch processing
   - Download corrected output

3. **Image OCR**
   - Image upload interface
   - OCR.space API integration (free tier)
   - Language: Arabic (ara)
   - Text extraction + correction

4. **Audio Upload**
   - Audio file upload (MP3, WAV, etc.)
   - Whisper speech-to-text transcription
   - Correction of transcribed text

5. **Live Recording**
   - Browser-based microphone access
   - streamlit-mic-recorder integration
   - Real-time transcription and correction

**Metrics Display:**

- Original vs Corrected comparison
- Character/Word accuracy
- Precision, Recall, F1-score
- Detailed character count statistics

---

## Build System

### LaTeX Paper Compilation

**Build Script:** `paper/combile.bat`

**Purpose:** Compile IEEE conference-format academic paper

**Toolchain Detection:**

1. **Primary:** latexmk (with XeLaTeX engine)
   - Full automation
   - Intelligent re-running
   - Bibliography and references
2. **Fallback:** XeLaTeX (3 passes)
   - Unicode support for Arabic
   - Direct PDF generation
3. **Last Resort:** PDFLaTeX (3 passes)
   - Basic LaTeX compilation

**Usage:**

```bash
cd paper
combile.bat paper.tex          # Compile paper.tex
combile.bat                    # Default: paper.tex
```

**Output:**

- `paper.pdf` - Final IEEE-formatted paper
- `paper.log` - Compilation log
- `paper.aux` - Bibliography and references
- `paper.synctex.gz` - Editor synchronization

**Paper Structure:**

```
- Title
- Authors (currently commented)
- Abstract
- Keywords
- Sections:
  1. Introduction
  2. Literature Review
  3. Methods (data, normalization, model)
  4. Experiments (metrics, protocol)
  5. Results and Discussion
  6. Conclusion
- Bibliography (23+ references)
- Figures (4 diagrams with detailed explanations)
```

**Recent Build Status:**

```
✓ XeLaTeX successful
✓ PDF generated (411+ KB)
✓ No fatal compilation errors
⚠ Minor: Underfull hbox warnings (formatting)
⚠ Minor: Font substitutions (Times New Roman fallback)
```

---

## File Descriptions

### Root Directory Files

#### `app.py` (Production Streamlit App)

- **Lines:** ~200
- **Purpose:** Multi-modal Arabic text correction interface
- **Key Classes:** None (functional programming)
- **Key Functions:**
  - `load_corrector()` - Load/cache AraBART model
  - `correct_text(text)` - Process text through model
  - `ocr_space_image()` - Call OCR API
  - `calculate_detailed_metrics()` - Compute evaluation metrics
  - `remove_punctuation()` - Text preprocessing
- **Dependencies:** transformers, streamlit, pillow, requests, whisper, jiwer
- **Input Formats:** Text, File, Image (URL), Audio
- **Output Formats:** Corrected text, metrics JSON, downloadable files

#### `README.md` (Project Overview)

- **Purpose:** User-facing project introduction
- **Sections:**
  - Project overview and features
  - Core functionality description
  - Technical implementation details
  - Input/output method documentation
  - Model architecture descriptions
  - Performance metrics
- **Audience:** General users, researchers, developers

#### `TECHNICAL_DOCUMENTATION.md` (Deep Technical Guide)

- **Purpose:** Comprehensive technical reference
- **Content:**
  - Detailed architecture explanations
  - Data pipeline specifics
  - Training protocol details
  - Model configuration
  - Performance benchmarks
  - Troubleshooting guides
- **Audience:** ML engineers, researchers, contributors

#### `paper.md` (Paper Markdown Version)

- **Purpose:** Research paper in Markdown format
- **Sections:**
  - Abstract and contributions
  - Related work
  - Methodology (data, noise injection, model)
  - Experimental setup
  - Results analysis
  - Error type taxonomy
- **Status:** Pre-LaTeX version

#### `requirements.txt` (Python Dependencies)

```
--extra-index-url https://download.pytorch.org/whl/cpu
streamlit
transformers==4.36.2
torch==2.7.1
jiwer
pillow
requests
openai-whisper
ffmpeg-python
streamlit-mic-recorder
pandas
plotly
soundfile
```

#### `packages.txt` (System Dependencies)

```
ffmpeg
```

#### `.python-version` (Python Version Specification)

- Specifies Python version for pyenv/virtual environments

#### `.venv/` (Virtual Environment)

- Isolated Python environment with all installed packages
- Created by: `python -m venv .venv`
- Activated by: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac)

#### `.git/` (Git Repository)

- Version control metadata
- Commit history
- Branch information

---

### Notebook Files

#### `NLP_Arabic_Text_Correction_Traditional.ipynb` (Custom Transformer)

- **Cells:** ~30-50
- **Purpose:** Train custom character-level seq2seq Transformer from scratch
- **Key Sections:**
  1. Setup and imports
  2. Data loading and preprocessing
  3. Arabic normalization
  4. Character tokenization
  5. Model architecture definition
  6. Training loop with scheduled sampling
  7. Evaluation and metrics computation
  8. Error analysis (CER, confusion matrix)
  9. Visualization of results
- **Output:** Trained model checkpoint, metrics plots, error analysis
- **Runtime:** ~2-3 hours on GPU, ~8-10 hours on CPU

#### `NLP_Project_LLM.ipynb` (Pre-trained Model Experiments)

- **Purpose:** Explore and evaluate CAMeL-Lab AraBART
- **Key Sections:**
  1. Model loading (HuggingFace)
  2. Inference examples
  3. Batch processing
  4. Metrics comparison with custom model
  5. Error analysis on pre-trained outputs
- **Output:** Comparative performance analysis

---

### Paper Directory

#### `paper.tex` (Main IEEE Conference Paper)

- **Lines:** ~400-500
- **Document Class:** IEEEtran (conference mode)
- **Sections:**
  1. **Title & Metadata**
     - Title: "Conference-Ready Arabic Text Correction..."
     - Authors: Currently commented
     - Abstract (150-200 words)
     - Keywords

  2. **Introduction**
     - Problem motivation
     - Contributions
     - Paper structure

  3. **Literature Review**
     - seq2seq architectures
     - Transformer models
     - Pre-training approaches
     - Arabic NLP specifics
     - QALB benchmarks
     - System-level integration

  4. **Methods**
     - Data pipeline (Youm7 crawling, filtering)
     - Normalization (Alef, Yaa, Hamza)
     - Noise injection (character & punctuation)
     - Custom Transformer architecture
     - Pre-trained LLM service description

  5. **Experiments**
     - Metrics definition (CER, WER, BLEU)
     - Training protocol details
     - Scheduled sampling decay

  6. **Results and Discussion**
     - Learning curves (38.38% → 91.69%)
     - Performance table (CER, training time, model size)
     - Error distribution analysis
     - Character confusion matrix interpretation

  7. **Conclusion**
     - Summary of contributions
     - Future work directions

  8. **Bibliography**
     - 23+ internet-sourced references
     - Authors, titles, venues, dates
     - URLs for online resources

- **Figures:**
  1. Transformer architecture diagram with encoder-decoder detail
  2. Encoder-decoder block layer composition
  3. CER distribution histogram across evaluation samples
  4. Character-level confusion matrix with Arabic letters

- **Tables:**
  1. Custom vs Pre-trained model comparison (CER, time, size, speed)

- **Build:** XeLaTeX → XDVIPDFmx → PDF (411+ KB)

#### `combile.bat` (Build Automation Script)

- **Purpose:** Compile paper.tex with intelligent toolchain fallback
- **Logic:**
  1. Check for latexmk availability
  2. If not: Fall back to XeLaTeX (3 passes)
  3. If not: Fall back to PDFLaTeX (3 passes)
  4. Error handling and logging
- **Features:**
  - Parameter support: `combile.bat filename.tex`
  - Default: `paper.tex`
  - Success/failure messaging
  - Exit codes for scripting

#### `dl_paper.tex` (IEEE Format Reference)

- **Purpose:** Template/reference for proper IEEE conference formatting
- **Use:** Reference for `paper.tex` structure and style

#### `paper.pdf` (Compiled Output)

- **Size:** 411+ KB
- **Pages:** 4
- **Format:** Letter size (8.5" × 11")
- **Content:** Complete academic paper as specified in `paper.tex`

#### `Diagrams/` (Figure Directory)

**Transfromer.jpg**

- Encoder-decoder data flow
- Input/output streams
- Attention mechanisms
- Vocabulary projection

**example.jpg**

- Detailed layer-by-layer architecture
- Self-attention, cross-attention blocks
- Residual connections
- Normalization points

**CER.png**

- Histogram of Character Error Rates
- Distribution across 1,000 test samples
- Central peak at low CER values
- Long-tail distribution

**CM.png**

- Confusion matrix heatmap
- Predicted vs reference Arabic characters
- Top frequent characters
- Diagonal dominance for correct predictions

---

### Auxiliary Files

#### `paper.aux` (LaTeX Auxiliary)

- Bibliography entries
- Cross-reference labels
- Page/section numbering

#### `paper.log` (Build Log)

- Compilation messages
- Font warnings
- Missing references
- Build duration

#### `paper.fdb_latexmk` (Latexmk Database)

- File modification tracking
- Dependency graph
- Rule application records

#### `paper.fls` (File List)

- All input files read during compilation
- All output files generated

#### `NLP Paper hour1.docx` (Original Document)

- Original Word format input
- Used for initial paper conversion to LaTeX

---

## Performance Summary

### Custom Transformer Model

| Metric                     | Value        |
| -------------------------- | ------------ |
| Training Accuracy (Peak)   | 91.69%       |
| Final Accuracy (Epoch 10)  | 89.36%       |
| Character Error Rate (CER) | 0.0364       |
| BLEU Score                 | 0.292        |
| Training Duration          | ~100 minutes |
| Model Size                 | ~50 MB       |
| Inference Speed            | Fast         |
| Data Used                  | 10,000 pairs |

### Pre-trained LLM (CAMeL-Lab AraBART)

| Metric                     | Value      |
| -------------------------- | ---------- |
| Character Error Rate (CER) | 0.0950     |
| Model Size                 | ~1.5 GB    |
| Fine-tuning Required       | No         |
| Inference Speed            | Moderate   |
| Grammatical Fluency        | Strong     |
| Deployment Readiness       | Production |

### Trade-offs

| Aspect                       | Custom Transformer | Pre-trained LLM |
| ---------------------------- | ------------------ | --------------- |
| **Accuracy**                 | 0.0364 CER         | 0.0950 CER      |
| **Size**                     | 50 MB              | 1.5 GB          |
| **Speed**                    | Fast               | Moderate        |
| **Training**                 | 100 min            | 0 min           |
| **Contextual Understanding** | Good               | Excellent       |
| **Grammatical Fluency**      | Good               | Excellent       |
| **Controllability**          | High               | Low             |
| **Interpretability**         | Medium             | Low             |

---

## Usage Examples

### Running the Streamlit Application

```bash
# Activate virtual environment
.venv\Scripts\activate

# Launch app
streamlit run app.py

# Access at http://localhost:8501
```

### Training Custom Model

```python
# In Jupyter/IPython or notebook
exec(open('NLP_Arabic_Text_Correction_Traditional.ipynb').read())

# Or run through notebook interface
```

### Compiling the Paper

```bash
cd paper
combile.bat                    # Compile paper.tex
combile.bat custom_name.tex    # Compile custom file
```

### Using the Models Programmatically

```python
# Pre-trained model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/arabart-qalb15-gec-ged-13")
model = AutoModelForSeq2SeqLM.from_pretrained("CAMeL-Lab/arabart-qalb15-gec-ged-13")
corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

result = corrector("نص عربي بها أخطاء", max_length=512)
print(result[0]['generated_text'])
```

---

## Key References (23+)

1. Sutskever et al. (2014) - Sequence to Sequence Learning with Neural Networks
2. Bahdanau et al. (2015) - Neural Machine Translation by Jointly Learning to Align
3. Vaswani et al. (2017) - Attention Is All You Need
4. Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers
5. Lewis et al. (2020) - BART: Denoising Sequence-to-Sequence Pre-training
6. Raffel et al. (2020) - T5: Exploring the Limits of Transfer Learning
7. Liu et al. (2020) - mBART: Multilingual Denoising Pre-training
8. Antoun et al. (2020) - AraBERT: Transformer-based Model for Arabic
9. Inoue et al. (2021) - The Interplay of Variant, Size, and Task Type (CAMEL)
10. Rozovskaya et al. (2014, 2015) - QALB Shared Task
11. Abandah et al. (2022) - BiLSTM-based Soft Spelling Correction
12. Qaraghuli & Jaafar (2024) - Arabic Soft Spelling Correction with T5
13. Kingma & Ba (2015) - Adam: A Method for Stochastic Optimization
14. Levenshtein (1966) - Binary Codes and Edit Distance
15. Papineni et al. (2002) - BLEU: Automatic Evaluation for Machine Translation
16. Radford et al. (2023) - Whisper: Robust Speech Recognition
17. Paszke et al. (2019) - PyTorch: Deep Learning Library
18. Wolf et al. (2020) - Transformers: HuggingFace Library
19. CAMeL Lab (2023) - arabart-qalb15-gec-ged-13 Model Card
20. OCR.space (2024) - Free OCR API Documentation
21. Streamlit (2024) - Web app framework
22. Morris (2024) - jiwer: Similarity Metrics for Speech Recognition
23. Additional papers on Arabic morphology, NLP benchmarks, and error analysis

---

## Future Enhancements

1. **Model Improvements**
   - Larger custom model variants
   - Domain-specific fine-tuning
   - Ensemble methods
   - Cross-lingual transfer

2. **Data Enhancements**
   - Expanded corpus from additional sources
   - Human-annotated error taxonomies
   - Dialect-specific models
   - Time-series analysis

3. **System Improvements**
   - Multi-GPU training support
   - Distributed inference
   - Model quantization
   - API service (FastAPI/Flask)

4. **Evaluation**
   - External benchmark validation (QALB)
   - Human evaluation protocols
   - Error taxonomy refinement
   - Confidence scoring

5. **Deployment**
   - Docker containerization
   - Cloud deployment (AWS/GCP)
   - Mobile app
   - Browser extension

---

## Troubleshooting

### Streamlit Issues

**Problem:** Model loading times out  
**Solution:** Increase cache timeout, pre-load models, use smaller batch sizes

**Problem:** Out of memory on GPU  
**Solution:** Reduce batch size, enable gradient checkpointing, use CPU fallback

### LaTeX Build Issues

**Problem:** "latexmk not found"  
**Solution:** Install MiKTeX or TeX Live, or run fallback in combile.bat

**Problem:** "Missing font: Times New Roman"  
**Solution:** Install fontspec package, switch to available system font

**Problem:** Arabic text not rendering  
**Solution:** Use XeLaTeX engine (default in combile.bat), ensure UTF-8 encoding

---

## Contact & Support

- **Project Type:** Academic Research + Production Deployment
- **Maintained By:** Arabic NLP Research Team (MSA University)
- **Status:** Active Development
- **Last Updated:** April 2026

---

**End of Documentation**
