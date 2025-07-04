import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from jiwer import cer
import requests
from PIL import Image
import io
import os
import whisper
import tempfile
from streamlit_mic_recorder import speech_to_text
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

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
    result = corrector(text, max_length=512, clean_up_tokenization_spaces=True)
    corrected_text = result[0]['generated_text']
    # Trim any "." from the beginning or end of the corrected text
    corrected_text = corrected_text.strip('.')
    return corrected_text

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
    ref_words = reference.strip().split()
    pred_words = prediction.strip().split()
    correct = sum(r == p for r, p in zip(ref_words, pred_words))
    total = max(len(ref_words), len(pred_words))
    if total == 0:
        return 0.0
    return correct / total

# Function to calculate detailed metrics
def calculate_detailed_metrics(reference, prediction):
    ref_words = reference.strip().split()
    pred_words = prediction.strip().split()
    
    # Word-level metrics
    correct_words = sum(r == p for r, p in zip(ref_words, pred_words))
    total_ref_words = len(ref_words)
    total_pred_words = len(pred_words)
    
    # Character-level metrics
    ref_chars = len(reference.replace(" ", ""))
    pred_chars = len(prediction.replace(" ", ""))
    correct_chars = sum(1 for r, p in zip(reference, prediction) if r == p)
    
    # Calculate precision, recall, F1
    precision = correct_words / total_pred_words if total_pred_words > 0 else 0
    recall = correct_words / total_ref_words if total_ref_words > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
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
        'total_pred_chars': pred_chars
    }

# Function to create word comparison dataframe
def create_word_comparison_df(reference, prediction):
    ref_words = reference.strip().split()
    pred_words = prediction.strip().split()
    
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
    ref_words = reference.strip().split()
    pred_words = prediction.strip().split()
    
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
    whisper_model = load_whisper()
    audio_file = st.file_uploader("Upload an audio file (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        st.audio(audio_file)
        if st.button("Transcribe Audio"):
            result = whisper_model.transcribe(tmp_path, language="ar")
            transcribed_text = result["text"]
            st.success("Transcribed Text:")
            st.write(transcribed_text)
            # Pass to correction model
            corrected = correct_text(transcribed_text)
            st.success("Corrected Text:")
            st.write(corrected)
            st.session_state['corrected'] = corrected

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
            cer_score = cer(reference_text, corrected_text)
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
                st.write(f"- Character accuracy: {detailed_metrics['correct_chars']/max(detailed_metrics['total_ref_chars'], detailed_metrics['total_pred_chars'])*100:.1f}%")
            
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
                    detailed_metrics['correct_chars']/max(detailed_metrics['total_ref_chars'], detailed_metrics['total_pred_chars'])*100
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