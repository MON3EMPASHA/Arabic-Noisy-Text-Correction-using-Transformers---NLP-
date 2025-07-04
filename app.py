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
    return result[0]['generated_text']

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

st.title("Arabic Spelling Correction App")
st.write("""
- Input Arabic text manually, upload a text file, or upload an image (OCR).
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
    input_text = st.text_area("Enter Arabic text with spelling mistakes:")
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
    st.header("Evaluate Correction (Optional)")
    eval_option = st.radio("Provide reference for evaluation:", ["Manual Reference Text", "Upload Reference File"], key="eval_option")
    reference_text = ""
    if eval_option == "Manual Reference Text":
        reference_text = st.text_area("Enter the correct reference text:", key="ref_text")
    else:
        ref_file = st.file_uploader("Upload the correct reference .txt file", type=["txt"], key="ref_file")
        if ref_file is not None:
            reference_text = ref_file.read().decode("utf-8")
            st.text_area("Reference File Content:", reference_text, height=200, disabled=True, key="ref_file_content")
    if st.button("Evaluate", key="evaluate"):
        if reference_text.strip():
            accuracy = calculate_accuracy(reference_text, st.session_state['corrected'])
            cer_score = cer(reference_text, st.session_state['corrected'])
            st.info(f"**Accuracy:** {accuracy*100:.2f}%")
            st.info(f"**Character Error Rate (CER):** {cer_score:.4f}")
        else:
            st.warning("Please provide the correct reference text or file.") 