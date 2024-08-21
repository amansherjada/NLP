import os
import tempfile
import whisper
import gradio as gr
from gtts import gTTS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Load Groq API key from environment variable
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

groq_client = Groq(api_key=groq_api_key)

# Attempt to load Whisper model
try:
    whisper_model = whisper.load_model("base")
except AttributeError:
    print("Error: The 'whisper' library does not support 'load_model'.")
    whisper_model = None

def process_audio(audio_file):
    if whisper_model is None:
        return "Whisper model could not be loaded.", None

    try:
        # Transcribe audio using Whisper
        result = whisper_model.transcribe(audio_file)
        user_text = result['text']

        # Generate response using Llama 8b model with Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": user_text}
            ],
            model="llama3-8b-8192",
        )
        response_text = chat_completion.choices[0].message.content

        # Convert response text to speech using gTTS
        tts = gTTS(text=response_text, lang='en')
        audio_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
        tts.save(audio_file_path)

        return response_text, audio_file_path
    except Exception as exp:
        return str(exp), None

webapp = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Response"), gr.Audio(label="Response Audio")],
    live=True,
    title="Speech to Speech",
    description="Upload or Record your Audio and get a response with audio playback.",
    theme=gr.themes.Monochrome()
)

if __name__ == "__main__":
    webapp.launch()