#pip install --upgrade gradio
import gradio as gr
from TTS.api import TTS
from datetime import datetime
import time
import traceback
import re
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import os
# Global configuration (TOP OF FILE)
MAX_FILE_SIZE_MB = 100  # 100MB limit
os.environ['GRADIO_MAX_FILE_SIZE'] = f"{MAX_FILE_SIZE_MB}MB"

LANGUAGE_MAPPINGS = {
    'ar': 'Arabic',
    'cs': 'Czech',
    'de': 'German',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'nl': 'Dutch',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'tr': 'Turkish',
    'zh': 'Chinese',
    'he': 'Hebrew'
}

class TTSVoiceCloningTool:
    def __init__(self):
        self.tts = None
        self.max_lengths = { 
            'fr': 273,
            'en': 300,
            'es': 280,
            'de': 275,
            'it': 270,
            'pt': 275,
            'hi': 260,
            'tr': 255,
            'ru': 265,
            'nl': 275,
            'cs': 270,
            'ar': 240,
            'zh': 200,
            'ja': 180,
            'ko': 180,
            'hu': 260,
            'he': 240
        }

    def load_tts_model(self):
        """Load the TTS model."""
        try:
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def unload_tts_model(self):
        """Unload the TTS model to free memory."""
        if self.tts is not None:
            del self.tts
            self.tts = None

    def get_max_length(self, language):
        """Get maximum text length for a language."""
        return self.max_lengths.get(language, 250)  # Default to 250 characters

    def split_text(self, text, language):
        """Split text into batches respecting sentence boundaries and character limits."""
        max_length = self.get_max_length(language)
        sentences = re.split(r'(?<=[.!?]) +', text)
        batches = []
        current_batch = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_length:
                if current_batch:
                    batches.append(' '.join(current_batch))
                    current_batch = []
                    current_length = 0
                
                # Handle sentences longer than max_length
                while sentence_length > max_length:
                    split_pos = sentence[:max_length].rfind(' ')
                    if split_pos == -1:
                        split_pos = max_length
                    batches.append(sentence[:split_pos])
                    sentence = sentence[split_pos:].lstrip()
                    sentence_length = len(sentence)

            if sentence:
                current_batch.append(sentence)
                current_length += sentence_length + 1  # +1 for space

        if current_batch:
            batches.append(' '.join(current_batch))

        return batches

    def generate_audio(self, text, output_path, target_language, input_path, speed):
        """Generate audio using TTS with batch processing."""
        print("Generating audio...")
        start_time = time.time()

        try:
            self.load_tts_model()
            batches = self.split_text(text, target_language)
            
            if not batches:
                raise ValueError("No text to process after splitting")

            temp_files = []
            for i, batch in enumerate(batches):
                print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} characters)")
                temp_path = f"temp_{i}_{datetime.now().strftime('%H%M%S')}.wav"
                
                self.tts.tts_to_file(
                    text=batch,
                    speaker_wav=input_path,
                    language=target_language,
                    file_path=temp_path,
                    speed=speed
                )
                temp_files.append(temp_path)

            # Combine audio files
            combined = AudioSegment.empty()
            for tf in temp_files:
                combined += AudioSegment.from_wav(tf)
            
            combined.export(output_path, format="mp3")  # Corrected format to mp3
            
            # Cleanup temp files
            for tf in temp_files:
                os.remove(tf)

            end_time = time.time()
            execution_time = (end_time - start_time) / 60
            print(f"Audio generated in {execution_time:.2f} minutes")
            return output_path

        except Exception as e:
            print(f"Generation error: {traceback.format_exc()}")
            raise
        finally:
            self.unload_tts_model()

def tts_interface(text, reference_audio, language, speed):
    """Wrapper for the TTS tool to integrate with Gradio."""
    if not reference_audio:
        return "Error: Please provide a reference audio file for voice cloning."

    try:
        audio_path = reference_audio[0] if isinstance(reference_audio, tuple) else reference_audio
        
        # 1. File existence check
        if not os.path.exists(audio_path):
            return "Error: File not found"
            
        # 2. Size validation
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        if file_size > MAX_FILE_SIZE_MB:
            return f"Error: File size {file_size:.1f}MB exceeds {MAX_FILE_SIZE_MB}MB limit"

        # 3. Duration validation
        audio = AudioSegment.from_file(audio_path)
        if len(audio) > 600 * 1000:  # 10 minutes
            return "Error: Audio exceeds 10 minute limit"
    except:
        return "Error: Invalid reference audio format"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"Tts_{timestamp}.mp3"  # Correct extension for MP3

    tts_tool = TTSVoiceCloningTool()
    try:
        # Removed unnecessary text length validation
        result_path = tts_tool.generate_audio(text, output_path, language, audio_path, speed)
        
        # Add file validation before returning
        if not os.path.exists(result_path):
            return "Error: Failed to generate audio file"
        if os.path.getsize(result_path) == 0:
            return "Error: Generated empty audio file"
            
        return result_path
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    tts_tool = TTSVoiceCloningTool()
    supported_languages = [
        (LANGUAGE_MAPPINGS[code], code) 
        for code in tts_tool.max_lengths.keys()
    ]
    with gr.Blocks() as demo:
        gr.Markdown("""
        # 🎙️ TTS Voice Cloning Tool  
        Convert long texts into speech using batch processing. Automatic text splitting by sentence boundaries.
        """)

        with gr.Row():
            text_input = gr.Textbox(
                label="Enter Text to Convert to Speech",
                placeholder="Paste your long text here (supports up to 10,000 characters)...",
                lines=8,
                max_lines=20,
                show_copy_button=True
            )

        with gr.Row():
            reference_audio_input = gr.Audio(
                label="Reference Audio File",
                sources=["upload"],
                type="filepath",
                max_length=600  # 10 minutes in seconds
            )
            language_input = gr.Dropdown(
                label="Target Language",
                choices=supported_languages,
                value="en"
            )
            speed_input = gr.Slider(
                label="Output Audio Speed",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1
            )

        gr.Markdown("**Note:** Long texts will be automatically split into batches for processing. Character limits vary by language.")
        output_audio = gr.Audio(label="Generated Audio", type="filepath")
        submit_button = gr.Button(value="🎧 Generate Audio", variant="primary")

        submit_button.click(
            tts_interface,
            inputs=[text_input, reference_audio_input, language_input, speed_input],
            outputs=[output_audio]
        )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7862
    )

if __name__ == "__main__":
    main()
