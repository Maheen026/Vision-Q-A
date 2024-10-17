import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO

# Load image captioning model from Hugging Face (BLIP)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate image caption
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to convert text to audio
def text_to_audio(text):
    tts = gTTS(text)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# Function to recognize speech from audio input
def recognize_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for audio input...")
        audio_data = r.listen(source)
        try:
            text = r.recognize_google(audio_data)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.write("Could not request results from Google Speech Recognition service.")
            return None

# Streamlit app interface
st.title("Image Audio Description Generator with Voice Interaction")
st.write("Upload an image to generate a description and audio. You can also communicate with the system using your voice.")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Step 1: Generate description using the BLIP model
    with st.spinner('Generating description...'):
        description = generate_caption(image)
    
    # Step 2: Convert description to speech using gTTS
    with st.spinner('Generating audio...'):
        audio_buffer = text_to_audio(description)

    # Display the generated description
    st.write(f"Generated Description: {description}")
    
    # Play the generated audio
    st.audio(audio_buffer, format="audio/mp3")

# Step 3: Voice input for user queries
if st.button("Speak to the system"):
    user_input = recognize_audio()
    if user_input:
        # Here you can add any custom handling for user input
        # For now, we can respond with the generated description
        response_audio = text_to_audio(f"You asked: {user_input}. The description of the image is: {description}.")
        st.audio(response_audio, format="audio/mp3")
