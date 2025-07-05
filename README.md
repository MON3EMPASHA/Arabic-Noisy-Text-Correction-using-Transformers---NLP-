# Arabic Text Correction Transformer Project

## 📋 Project Overview

This project implements an advanced Arabic text correction system using two different approaches:

1. **Traditional Transformer Architecture** - A custom-built sequence-to-sequence transformer model trained from scratch
2. **Pre-trained LLM Approach** - Using the CAMeL-Lab Arabic BART model for text correction

GUI : **Web Application** - A comprehensive Streamlit-based interface for text correction with multiple input methods using the Pre-trained LLM Approach

The system is designed to correct spelling mistakes, grammatical errors, and text normalization issues in Arabic text, making it particularly useful for processing noisy Arabic text from various sources.

## 🚀 Features

### Core Functionality
- **Arabic Text Correction**: Corrects spelling and grammatical errors in Arabic text
- **Multiple Input Methods**: Manual text input, file upload, OCR from images, speech-to-text
- **Comprehensive Evaluation**: Character Error Rate (CER), Word Error Rate (WER), BLEU scores
- **Error Analysis**: Detailed breakdown of error types and patterns
- **Real-time Processing**: Live audio recording and transcription

### Input Methods
1. **Manual Text Input**: Direct text entry with example templates
2. **File Upload**: Support for .txt files
3. **Image OCR**: Extract text from images using OCR.space API
4. **Audio Upload**: Speech-to-text conversion using Whisper
5. **Live Recording**: Real-time audio recording and transcription

## 🔧 Technical Implementation

### 1. Traditional Transformer Approach (`NLP_Arabic_Text_Correction_Traditional.ipynb`)

#### Architecture Overview
The traditional approach implements a custom transformer model with the following components:

```python
# Model Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_SEQ_LEN = 128
EMBED_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.1
```

#### Key Components

**1. Text Normalization**
```python
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)  # Normalize Alef variants
    text = re.sub("ى", "ي", text)      # Normalize Yaa variants
    text = re.sub("ؤ", "و", text)      # Normalize Waw variants
    text = re.sub("ئ", "ي", text)      # Normalize Hamza
    text = re.sub("ة", "ه", text)      # Normalize Taa Marbouta
    text = re.sub("[ًٌٍَُِّْ]", "", text)  # Remove diacritics
    text = re.sub("ـ", "", text)       # Remove tatweel
    return text
```

**2. Character-Level Tokenization**
- Builds vocabulary from all unique characters in the dataset
- Includes special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Vocabulary size: ~304 characters

**3. Transformer Model Architecture**
```python
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.positional_encoding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, EMBED_DIM))
        
        # Encoder: Processes noisy input text
        encoder_layer = nn.TransformerEncoderLayer(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)
        
        # Decoder: Generates corrected text
        decoder_layer = nn.TransformerDecoderLayer(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_DECODER_LAYERS)
        
        # Output layer: Converts to character probabilities
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)
```

**4. Training Features**
- **Scheduled Sampling**: Gradually transitions from teacher forcing to using model predictions
- **Mixed Precision Training**: Uses CUDA AMP for faster training
- **Checkpointing**: Saves model state after each epoch
- **Resume Training**: Can continue from previous checkpoints

#### Training Process
1. **Data Loading**: 10,000 noisy-clean text pairs
2. **Preprocessing**: Text normalization and character-level encoding
3. **Training Loop**: 10 epochs with scheduled sampling
4. **Evaluation**: Real-time accuracy and loss monitoring

#### Performance Metrics
- **Training Accuracy**: ~91.5% after 10 epochs
- **Character Error Rate (CER)**: ~0.0364 on test samples
- **BLEU Score**: ~0.292 average

### Model Architecture Details

The traditional transformer implements a sequence-to-sequence architecture specifically designed for Arabic text correction:

#### 1. Embedding Layer
```python
self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
```
- **Purpose**: Converts character indices to dense vector representations
- **Dimensions**: `vocab_size × EMBED_DIM` (304 × 256)
- **Initialization**: Random initialization with learned parameters

#### 2. Positional Encoding
```python
self.positional_encoding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, EMBED_DIM))
```
- **Purpose**: Provides position information to the transformer
- **Method**: Learnable positional encodings (not sinusoidal)
- **Dimensions**: `1 × MAX_SEQ_LEN × EMBED_DIM` (1 × 128 × 256)

#### 3. Encoder Architecture
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=EMBED_DIM,           # 256
    nhead=NUM_HEADS,             # 8
    dim_feedforward=FF_DIM,      # 512
    dropout=DROPOUT,             # 0.1
    batch_first=True
)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)
```

**Encoder Layer Components:**
- **Multi-Head Self-Attention**: 8 attention heads with 32-dimensional attention each
- **Feed-Forward Network**: Two linear layers with ReLU activation
- **Layer Normalization**: Applied before each sub-layer
- **Residual Connections**: Around each sub-layer
- **Dropout**: 10% dropout for regularization

#### 4. Decoder Architecture
```python
decoder_layer = nn.TransformerDecoderLayer(
    d_model=EMBED_DIM,           # 256
    nhead=NUM_HEADS,             # 8
    dim_feedforward=FF_DIM,      # 512
    dropout=DROPOUT,             # 0.1
    batch_first=True
)
self.encoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_DECODER_LAYERS)
```

**Decoder Layer Components:**
- **Masked Multi-Head Self-Attention**: Prevents looking at future tokens
- **Cross-Attention**: Attends to encoder output
- **Feed-Forward Network**: Same as encoder
- **Layer Normalization**: Applied before each sub-layer
- **Residual Connections**: Around each sub-layer

#### 5. Output Layer
```python
self.fc_out = nn.Linear(EMBED_DIM, vocab_size)
```
- **Purpose**: Projects decoder output to vocabulary space
- **Activation**: Linear (no activation function)
- **Output**: Logits for each character in vocabulary

### Forward Pass Algorithm

```python
def forward(self, src, tgt):
    # 1. Embedding and Positional Encoding
    src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1)]
    tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1)]
    
    # 2. Encoder Processing
    memory = self.encoder(src_emb)
    
    # 3. Decoder Processing
    output = self.decoder(tgt_emb, memory)
    
    # 4. Output Projection
    logits = self.fc_out(output)
    return logits
```


### 2. LLM-Based Approach (`NLP_Project_LLM.ipynb`)

#### Model Details
- **Model**: `CAMeL-Lab/arabart-qalb15-gec-ged-13`
- **Architecture**: Arabic BART (Bidirectional and Auto-Regressive Transformers)
- **Purpose**: Grammar Error Correction (GEC) and Grammar Error Detection (GED)

#### Implementation
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "CAMeL-Lab/arabart-qalb15-gec-ged-13"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
```

#### Advantages
- **Pre-trained**: Leverages large-scale Arabic text training
- **Specialized**: Specifically designed for Arabic grammar correction
- **Efficient**: No training required, ready to use
- **Robust**: Handles complex grammatical structures

### 3. Web Application (`app.py`)

#### Features

**1. Multiple Input Methods**
```python
option = st.radio(
    "Choose input method:", 
    [
        "Manual Text", 
        "Upload Text File", 
        "Upload Image (OCR)",
        "Upload Audio (Speech-to-Text)",
        "Record Audio (Live)"
    ]
)
```

**2. OCR Integration**
- Uses OCR.space API for image text extraction
- Supports JPG, JPEG, PNG formats
- Automatic Arabic language detection

**3. Speech-to-Text**
- **File Upload**: Whisper model for audio transcription
- **Live Recording**: Real-time audio capture and transcription
- **Language Support**: Optimized for Arabic speech

**4. Comprehensive Evaluation**
```python
def calculate_detailed_metrics(reference, prediction):
    return {
        'word_accuracy': correct_words / max(total_ref_words, total_pred_words),
        'precision': correct_words / total_pred_words,
        'recall': correct_words / total_ref_words,
        'f1_score': 2 * (precision * recall) / (precision + recall),
        'correct_words': correct_words,
        'total_ref_words': total_ref_words,
        'total_pred_words': total_pred_words,
        'correct_chars': correct_chars,
        'total_ref_chars': ref_chars,
        'total_pred_chars': pred_chars
    }
```

**5. Visualization and Analysis**
- **Performance Metrics**: Bar charts and pie charts
- **Word-by-Word Comparison**: Detailed analysis table
- **Error Pattern Analysis**: Error type distribution
- **Real-time Metrics**: Live accuracy and CER calculation

## 📊 Evaluation Results

### Traditional Transformer Performance
- **Training Loss**: Decreased from 2.28 to 0.43 over 10 epochs
- **Training Accuracy**: Reached 91.5% by epoch 4
- **Character Error Rate**: 0.0364 (3.64% error rate)
- **BLEU Score**: 0.292 average

### Error Analysis
**Top Character Confusions:**
1. Space ↔ ل (65 errors)
2. ا ↔ Space (63 errors)
3. ل ↔ Space (53 errors)
4. Space ↔ ا (52 errors)
5. ه ↔ ا (41 errors)

**Error Type Distribution:**
- **Substitutions**: Most common error type
- **Insertions**: Character additions
- **Deletions**: Character omissions

### LLM Approach Performance
- **Character Error Rate**: 0.095 (9.5% error rate)
- **Advantages**: Better handling of complex grammatical structures
- **Disadvantages**: Higher computational requirements



## 🎯 Usage Examples

### 1. Manual Text Correction
```python
# Example noisy text
noisy_text = "فذ اطار حرص كلصة الصيدلة"
corrected_text = correct_text(noisy_text)
# Output: "في اطار حرص كلية الصيدلة"
```

### 2. File Processing
```python
# Upload text file and process
with open('noisy_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
corrected = correct_text(text)
```

### 3. Image OCR
```python
# Extract text from image and correct
image_bytes = uploaded_image.getvalue()
extracted_text, error = ocr_space_image(image_bytes)
if extracted_text:
    corrected = correct_text(extracted_text)
```

### 4. Audio Processing
```python
# Transcribe audio and correct
whisper_model = whisper.load_model("base")
result = whisper_model.transcribe(audio_path, language="ar")
transcribed_text = result["text"]
corrected = correct_text(transcribed_text)
```

## 📈 Performance Optimization

### Training Optimizations
1. **Mixed Precision Training**: Reduces memory usage and speeds up training
2. **Scheduled Sampling**: Improves convergence and reduces exposure bias
3. **Checkpointing**: Enables training resumption and model recovery
4. **Batch Processing**: Efficient GPU utilization

### Inference Optimizations
1. **Model Caching**: Streamlit cache for model loading
2. **Batch Processing**: Process multiple texts simultaneously
3. **Memory Management**: Efficient tensor operations

## 🔍 Error Analysis and Debugging


### Research Directions
1. **Context-Aware Correction**: Understanding context for better corrections
2. **Dialect-Specific Models**: Region-specific Arabic dialect handling
3. **Domain Adaptation**: Specialized models for different text domains
4. **Ensemble Methods**: Combining multiple models for better performance

## 👥 Team Information

**Project Team**: SHAWARMA RETRIEVERS  
**Supervisors**: 
- Dr. Wael Goma
- Dr. Ahmed Mostafa

**Course**: Natural Language Processing (NLP)  
**Institution**: MSA

## 📄 License

This project is developed for educational purposes as part of the NLP course curriculum.

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome. Please contact the project team for any questions or contributions.

---

**Note**: This documentation provides a comprehensive overview of the Arabic Text Correction Transformer project. For specific implementation details, refer to the individual notebook files and the web application code. 