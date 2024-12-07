import gradio as gr
from TTS.api import TTS
from datetime import datetime
import time

class TTSVoiceCloningTool:
    def __init__(self):
        self.tts = None

    def load_tts_model(self):
        """Load the TTS model."""
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    def unload_tts_model(self):
        """Unload the TTS model to free memory."""
        del self.tts
        self.tts = None

    def get_supported_languages(self):
        """Return a list of supported languages."""
        self.load_tts_model()
        languages = self.tts.list_languages()
        self.unload_tts_model()
        return languages

    def generate_audio(self, text, output_path, target_language, input_path, speed):
        """Generate audio using TTS."""
        print("Generating audio...")
        start_time = time.time()

        self.load_tts_model()

        # Generate audio with adjustable speed
        self.tts.tts_to_file(
            text=text,
            speaker_wav=input_path,
            language=target_language,
            file_path=output_path,
            speed=speed  # Pass the speed parameter
        )

        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        print(f"Audio generated in {execution_time:.2f} minutes")

        self.unload_tts_model()
        return output_path

def tts_interface(text, reference_audio, language, speed):
    """Wrapper for the TTS tool to integrate with Gradio."""
    if not reference_audio:
        return "Error: Please provide a reference audio file for voice cloning."

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"Tts_{timestamp}.mp3"

    tts_tool = TTSVoiceCloningTool()
    try:
        result_path = tts_tool.generate_audio(text, output_path, language, reference_audio, speed)
        return result_path
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio GUI
def main():
    # Manually specify supported languages
    supported_languages = ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja"]  # Add or remove languages as necessary

    with gr.Blocks() as demo:
        gr.Markdown("""
        # 🎙️ TTS Voice Cloning Tool  
        Convert your text into speech using voice cloning! Provide a reference audio to mimic the voice, select the target language, and adjust the output speed.
        """)

        with gr.Row():
            text_input = gr.Textbox(
                label="Enter Text to Convert to Speech",
                placeholder="Type or paste the text you want to convert to speech here...",
                lines=8,
                max_lines=20
            )

        with gr.Row():
            reference_audio_input = gr.Audio(label="Reference Audio File", type="filepath")
            language_input = gr.Dropdown(
                label="Target Language",
                choices=supported_languages,
                value="en"  # Default to English
            )
            speed_input = gr.Slider(
                label="Output Audio Speed",
                minimum=0.5,
                maximum=2.0,
                value=1.0,  # Default speed
                step=0.1
            )

        output_audio = gr.Audio(label="Generated Audio", type="filepath")

        submit_button = gr.Button(value="🎧 Generate Audio", variant="primary")

        submit_button.click(
            tts_interface,
            inputs=[text_input, reference_audio_input, language_input, speed_input],
            outputs=[output_audio]
        )

    demo.launch(server_name="127.0.0.2", server_port=7862)

if __name__ == "__main__":
    main()
