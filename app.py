import streamlit as st
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from jiwer import cer
import requests
from PIL import Image
import io
import os
# import whisper
import tempfile
from streamlit_mic_recorder import speech_to_text
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import unicodedata
import soundfile as sf

# Function to remove specific punctuation marks
def remove_punctuation(text):
    # Normalize text to standard form
    text = unicodedata.normalize("NFKC", text)
    # Remove all Unicode punctuation and symbols
    text = ''.join(
        ch for ch in text
        if not unicodedata.category(ch).startswith(('P', 'S'))
    )
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load model and tokenizer once
@st.cache_resource
def load_corrector():
    model_name = "CAMeL-Lab/arabart-qalb15-gec-ged-13"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return corrector

corrector = load_corrector()

# Function to correct text
def correct_text(text):
    # Remove punctuation before correction
    cleaned_text = remove_punctuation(text)
    if not cleaned_text.strip():
        return ""
    global corrector
    tokenizer = getattr(corrector, 'tokenizer', None)
    outputs = []
    if tokenizer is not None:
        tokens = tokenizer.encode(cleaned_text)
        chunk_size = 512
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            result = corrector(chunk_text, max_length=512, clean_up_tokenization_spaces=True)
            if isinstance(result, list) and 'generated_text' in result[0]:
                corrected_chunk = result[0]['generated_text']
            else:
                corrected_chunk = str(result)
            outputs.append(corrected_chunk.strip('.') if isinstance(corrected_chunk, str) else corrected_chunk)
        return ''.join(outputs)
    else:
        # Fallback: just chunk by 512 characters
        for i in range(0, len(cleaned_text), 512):
            chunk = cleaned_text[i:i+512]
            result = corrector(chunk, max_length=512, clean_up_tokenization_spaces=True)
            if isinstance(result, list) and 'generated_text' in result[0]:
                corrected_chunk = result[0]['generated_text']
            else:
                corrected_chunk = str(result)
            outputs.append(corrected_chunk.strip('.') if isinstance(corrected_chunk, str) else corrected_chunk)
        return ''.join(outputs)

# Function to call OCR.space API
def ocr_space_image(image_bytes, api_key='helloworld', language='ara'):
    url_api = 'https://api.ocr.space/parse/image'
    # Provide a filename and content type!
    files = {'file': ('image.png', image_bytes, 'image/png')}
    data = {'apikey': api_key, 'language': language, 'isOverlayRequired': False}
    response = requests.post(url_api, files=files, data=data)
    result = response.json()
    if result.get('IsErroredOnProcessing'):
        return None, result.get('ErrorMessage', 'Unknown error')
    parsed_results = result.get('ParsedResults')
    if parsed_results:
        return parsed_results[0]['ParsedText'], None
    return None, 'No text found.'

# Function to calculate accuracy (word-level)
def calculate_accuracy(reference, prediction):
    # Remove punctuation from both reference and prediction for evaluation
    ref_clean = remove_punctuation(reference.strip())
    pred_clean = remove_punctuation(prediction.strip())
    ref_words = ref_clean.split()
    pred_words = pred_clean.split()
    correct = sum(r == p for r, p in zip(ref_words, pred_words))
    total = max(len(ref_words), len(pred_words))
    if total == 0:
        return 0.0
    return correct / total

# Function to calculate detailed metrics
def calculate_detailed_metrics(reference, prediction):
    # Remove punctuation from both reference and prediction for evaluation
    ref_clean = remove_punctuation(reference.strip())
    pred_clean = remove_punctuation(prediction.strip())
    
    ref_words = ref_clean.split()
    pred_words = pred_clean.split()
    
    # Word-level metrics
    correct_words = sum(r == p for r, p in zip(ref_words, pred_words))
    total_ref_words = len(ref_words)
    total_pred_words = len(pred_words)
    
    # Character-level metrics (without spaces)
    ref_chars = len(ref_clean.replace(" ", ""))
    pred_chars = len(pred_clean.replace(" ", ""))
    
    # Calculate correct characters by comparing character by character
    min_len = min(len(ref_clean), len(pred_clean))
    correct_chars = sum(1 for i in range(min_len) if ref_clean[i] == pred_clean[i])
    
    # Calculate precision, recall, F1
    precision = correct_words / total_pred_words if total_pred_words > 0 else 0
    recall = correct_words / total_ref_words if total_ref_words > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate character accuracy (ensure it doesn't exceed 100%)
    char_accuracy = correct_chars / max(ref_chars, pred_chars) if max(ref_chars, pred_chars) > 0 else 0
    char_accuracy = min(char_accuracy, 1.0)  # Cap at 100%
    
    return {
        'word_accuracy': correct_words / max(total_ref_words, total_pred_words) if max(total_ref_words, total_pred_words) > 0 else 0,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'correct_words': correct_words,
        'total_ref_words': total_ref_words,
        'total_pred_words': total_pred_words,
        'correct_chars': correct_chars,
        'total_ref_chars': ref_chars,
        'total_pred_chars': pred_chars,
        'char_accuracy': char_accuracy
    }

# Function to create word comparison dataframe
def create_word_comparison_df(reference, prediction):
    # Remove punctuation from both reference and prediction
    ref_clean = remove_punctuation(reference.strip())
    pred_clean = remove_punctuation(prediction.strip())
    
    ref_words = ref_clean.split()
    pred_words = pred_clean.split()
    
    max_len = max(len(ref_words), len(pred_words))
    ref_words.extend([''] * (max_len - len(ref_words)))
    pred_words.extend([''] * (max_len - len(pred_words)))
    
    df = pd.DataFrame({
        'Position': range(1, max_len + 1),
        'Reference': ref_words,
        'Prediction': pred_words,
        'Match': [r == p for r, p in zip(ref_words, pred_words)]
    })
    
    return df

# Function to analyze error patterns
def analyze_error_patterns(reference, prediction):
    # Remove punctuation from both reference and prediction
    ref_clean = remove_punctuation(reference.strip())
    pred_clean = remove_punctuation(prediction.strip())
    
    ref_words = ref_clean.split()
    pred_words = pred_clean.split()
    
    errors = []
    for i, (ref, pred) in enumerate(zip(ref_words, pred_words)):
        if ref != pred:
            errors.append({
                'position': i + 1,
                'reference': ref,
                'prediction': pred,
                'error_type': 'substitution' if len(ref) == len(pred) else 'length_mismatch'
            })
    
    return errors

def is_valid_audio(file_path):
    try:
        with sf.SoundFile(file_path) as f:
            return len(f) > 0
    except Exception:
        return False

st.title("Arabic Noisy Text Correction App")
st.write("""
- Input Arabic text manually, upload a text file, upload an image (OCR), upload an audio file (Speech-to-Text), or record audio (Live).
- The model will correct spelling/grammar mistakes.
- Optionally, provide the correct text or file to evaluate accuracy and CER.
""")

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

input_text = ""
ocr_error = None

if option == "Manual Text":
    # Noisy text examples for manual input
    noisy_examples = {
        "Example 1 -  مرحبا بكم في هذا التطبيف الجميل جدا": ".مرحبا بكم في هذا التطبيف الجميل جدا",
        "Example 2 - انا دهبت الي المدرسه في الصباح الباكر  ": ".انا دهبت الي المدرسه في الصباح الباكر",
        "Example 3 - هذو الكتاب دميل ددا و مفيد للقراءه و التعلم ": ".هذو الكتاب دميل ددا و مفيد للقراءه و التعلم"
    }
    
    # Option to use examples or manual input
    input_method = st.radio("Choose input method:", ["Enter text manually", "Use example text"], key="input_method")
    
    if input_method == "Enter text manually":
        input_text = st.text_area("Enter Arabic text with spelling mistakes:")
    else:
        example_choice = st.selectbox("Select an example:", list(noisy_examples.keys()))
        input_text = noisy_examples[example_choice]
        st.text_area("Selected example text:", input_text, height=100, disabled=True)
    
    if st.button("Correct Text", key="manual_correct"):
        if input_text.strip():
            corrected = correct_text(input_text)
            st.success("Corrected Text:")
            st.write(corrected)
            st.session_state['corrected'] = corrected
        else:
            st.warning("Please enter some text.")

elif option == "Upload Text File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        file_text = uploaded_file.read().decode("utf-8")
        st.text_area("File Content:", file_text, height=200, disabled=True)
        if st.button("Correct File", key="file_correct"):
            corrected = correct_text(file_text)
            st.success("Corrected Text:")
            st.write(corrected)
            st.session_state['corrected'] = corrected

elif option == "Upload Image (OCR)":
    uploaded_image = st.file_uploader("Upload an image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Extract and Correct Text", key="ocr_correct"):
            image_bytes = uploaded_image.getvalue()
            extracted_text, ocr_error = ocr_space_image(image_bytes)
            if extracted_text:
                st.text_area("Extracted Text:", extracted_text, height=200, disabled=True)
                corrected = correct_text(extracted_text)
                st.success("Corrected Text:")
                st.write(corrected)
                st.session_state['corrected'] = corrected
            else:
                st.error(f"OCR Error: {ocr_error}")

elif option == "Upload Audio (Speech-to-Text)":
    @st.cache_resource
    def load_whisper():
        return whisper.load_model("base")
    # Update the file uploader to support .ogg files
    uploaded_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg"])

    if uploaded_audio is not None:
        import tempfile
        import os
        # Save the uploaded file to a temporary file with the correct extension
        suffix = "." + uploaded_audio.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_audio.read())
            tmp_path = tmp_file.name

        # Check if file exists before passing to Whisper
        if os.path.exists(tmp_path):
            if is_valid_audio(tmp_path):
                whisper_model = load_whisper()
                result = whisper_model.transcribe(tmp_path, language="ar")
                transcribed_text = result["text"]
                st.write("Transcribed Text:", transcribed_text)
                # Pass to correction model
                corrected = correct_text(transcribed_text)
                st.success("Corrected Text:")
                st.write(corrected)
                st.session_state['corrected'] = corrected
            else:
                st.error("Uploaded audio file is empty or invalid. Please upload a valid audio file.")
        else:
            st.error("Audio file could not be saved. Please try again.")

elif option == "Record Audio (Live)":
    st.write("Record your voice and transcribe it to text.")
    st.write("Click the microphone icon below and speak:")
    
    text = speech_to_text(language='ar', use_container_width=True, just_once=True)
    
    if text:
        st.success("Transcribed Text:")
        st.write(text)
        # Pass to correction model
        corrected = correct_text(text)
        st.success("Corrected Text:")
        st.write(corrected)
        st.session_state['corrected'] = corrected

# Evaluation Section
if 'corrected' in st.session_state:
    st.markdown("---")
    st.header("📊 Evaluation & Analysis")
    
    # Example reference texts for demonstration
    example_texts = {
        "Example 1 - مرحبا بكم في هذا التطبيف الجميل جدا": {
            "input": "مرحبا بكم في هذا التطبيف الجميل جدا",
            "reference": "مرحباً بكم في هذا التطبيق الجميل جداً"
        },
        "Example 2 - انا دهبت الي المدرسه في الصباح الباكر": {
            "input": "انا دهبت الي المدرسه في الصباح الباكر",
            "reference": "أنا ذهبت إلى المدرسة في الصباح الباكر"
        },
        "Example 3 - هذو الكتاب دميل ددا و مفيد للقراءه و التعلم": {
            "input": "هذو الكتاب دميل ددا و مفيد للقراءه و التعلم",
            "reference": "هذا الكتاب جميل جداً ومفيد للقراءة والتعلم"
        }
    }
    
    eval_option = st.radio("Choose evaluation method:", 
                          ["Manual Reference Text", "Upload Reference File", "Use Example"], 
                          key="eval_option")
    
    reference_text = ""
    if eval_option == "Manual Reference Text":
        reference_text = st.text_area("Enter the correct reference text:", key="ref_text")
    elif eval_option == "Upload Reference File":
        ref_file = st.file_uploader("Upload the correct reference .txt file", type=["txt"], key="ref_file")
        if ref_file is not None:
            reference_text = ref_file.read().decode("utf-8")
            st.text_area("Reference File Content:", reference_text, height=200, disabled=True, key="ref_file_content")
    else:  # Use Example
        example_choice = st.selectbox("Select an example:", list(example_texts.keys()))
        if example_choice:
            example = example_texts[example_choice]
            st.info(f"**Original Text:** {example['input']}")
            st.info(f"**Reference:** {example['reference']}")
            reference_text = example['reference']
            # Use the corrected text from the model as input for evaluation
            corrected_text = correct_text(example['input'])
            st.info(f"**Model Output:** {corrected_text}")
            st.session_state['corrected'] = corrected_text
    
    if st.button("Evaluate & Analyze", key="evaluate"):
        if reference_text.strip():
            corrected_text = st.session_state['corrected']
            
            # Calculate basic metrics
            accuracy = calculate_accuracy(reference_text, corrected_text)
            # Remove punctuation from both reference and prediction for CER calculation
            ref_clean = remove_punctuation(reference_text.strip())
            pred_clean = remove_punctuation(corrected_text.strip())
            cer_score = cer(ref_clean, pred_clean)
            detailed_metrics = calculate_detailed_metrics(reference_text, corrected_text)
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Word Accuracy", f"{accuracy*100:.1f}%")
            with col2:
                st.metric("CER Score", f"{cer_score:.4f}")
            with col3:
                st.metric("Precision", f"{detailed_metrics['precision']*100:.1f}%")
            with col4:
                st.metric("F1 Score", f"{detailed_metrics['f1_score']*100:.1f}%")
            
            # Detailed metrics section
            st.subheader("📈 Detailed Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Word-level Analysis:**")
                st.write(f"- Correct words: {detailed_metrics['correct_words']}")
                st.write(f"- Total reference words: {detailed_metrics['total_ref_words']}")
                st.write(f"- Total prediction words: {detailed_metrics['total_pred_words']}")
                st.write(f"- Recall: {detailed_metrics['recall']*100:.1f}%")
            
            with col2:
                st.write("**Character-level Analysis:**")
                st.write(f"- Correct characters: {detailed_metrics['correct_chars']}")
                st.write(f"- Total reference characters: {detailed_metrics['total_ref_chars']}")
                st.write(f"- Total prediction characters: {detailed_metrics['total_pred_chars']}")
                st.write(f"- Character accuracy: {detailed_metrics['char_accuracy']*100:.1f}%")
            
            # Word comparison table
            st.subheader("🔍 Word-by-Word Comparison")
            comparison_df = create_word_comparison_df(reference_text, corrected_text)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Error analysis
            st.subheader("❌ Error Analysis")
            errors = analyze_error_patterns(reference_text, corrected_text)
            if errors:
                error_df = pd.DataFrame(errors)
                st.dataframe(error_df, use_container_width=True)
                
                # Error type distribution
                error_types = Counter([error['error_type'] for error in errors])
                if error_types:
                    fig = px.pie(values=list(error_types.values()), 
                               names=list(error_types.keys()),
                               title="Error Type Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("🎉 No errors found! Perfect match!")
            
            # Performance visualization
            st.subheader("📊 Performance Visualization")
            
            # Metrics bar chart
            metrics_data = {
                'Metric': ['Word Accuracy', 'Precision', 'Recall', 'F1 Score', 'Character Accuracy'],
                'Value': [
                    accuracy * 100,
                    detailed_metrics['precision'] * 100,
                    detailed_metrics['recall'] * 100,
                    detailed_metrics['f1_score'] * 100,
                    detailed_metrics['char_accuracy'] * 100
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            fig = px.bar(metrics_df, x='Metric', y='Value', 
                        title="Performance Metrics Comparison",
                        color='Value',
                        color_continuous_scale='RdYlGn')
            fig.update_layout(yaxis_title="Percentage (%)")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Please provide the correct reference text or file.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: white; padding: 20px;">
        Made by <b>SHAWARMA RETRIEVERS</b> | NLP <br>
        Supervised by <i>Dr. Wael Goma</i> & <i>Dr. Ahmed Mostafa</i>
    </div>
    """,
    unsafe_allow_html=True
) 