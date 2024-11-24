import gradio as gr
import os
import subprocess
import threading
import webbrowser
from pydub import AudioSegment
from pydub.utils import mediainfo
from OpenTranslator.translator import CustomTranslator
import unicodedata
import librosa

current_dir = os.path.dirname(os.path.abspath(__file__))
# Initialize the translator instance with an output directory
output_dir = os.path.join(current_dir, "output")

translator_instance = CustomTranslator(output_dir=output_dir)

# Define the languages dictionary
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Turkish": "tr",
    "Arabic": "ar",
    "Russian": "ru",
    "Hebrew": "he",
    "Hindi": "hi",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese (Mandarin)": "zh",
    "Czech": "cs",
    "Dutch": "nl",
    "Polish": "pl"
}

language_choices = [(lang, code) for lang, code in languages.items()]

# Define the translation options
TextTranslationOption = ["Local"]

# Function to handle file uploads
def upload_file(file):
    global audio_path
    audio_path = file.name
    #return f"Selected File Title: {os.path.basename(audio_path)}"

# Function to run the translation process
def run_translation(translation_method, target_lang):
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_translated.mp3")
    input_file = audio_path
    print(audio_path)
    input_duration = get_audio_duration(input_file)
    print('input_duration: '+str(input_duration))
    if input_duration > 30:
        max_chunk_duration = 30
        num_chunks = int(input_duration / max_chunk_duration)
        chunk_files = []
        Translation_chunk_files = []
        translated_text = []
        
        for chunk_idx in range(num_chunks):
            print('duration more then 30- num_chunks: '+str(num_chunks))
            print('duration more then 30- chunk_idx'+str(chunk_idx))
            start_time = chunk_idx * max_chunk_duration
            end_time = min((chunk_idx + 1) * max_chunk_duration, input_duration)
            chunk_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_chunk{chunk_idx + 1}.wav")
            
            split_audio_chunk(input_file, chunk_output_path, start_time, end_time)
            
            try:
                translation_result = translator_instance.process_audio_chunk(chunk_output_path,
                                                                             target_lang,
                                                                             chunk_idx, output_path, translation_method)
            except Exception as e:
                print(f"{e}")
                return "An Error occurred!"
            
            translated_text.append(translation_result)    

            chunk_files.append(chunk_output_path)
            Translation_chunk_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_Translation_chunk{chunk_idx + 1}.wav")
            
            Translation_chunk_files.append(Translation_chunk_output_path)

        final_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}-temp.wav")
        
        if translation_method == 'Local':
            merge_audio_files(Translation_chunk_files, final_output_path)

        subprocess.run(['ffmpeg', '-i', final_output_path, '-codec:a', 'libmp3lame', output_path], check=True)
        os.remove(final_output_path)

        delete_chunk_files(chunk_files)
        delete_chunk_files(Translation_chunk_files)
        chunk_files = []  # List to store individual chunk files
        Translation_chunk_files = []
        
        translation_result = ', '.join(translated_text)
        return translation_result, output_path

    if input_duration <= 30 and translation_method == 'Local':
        #translated_text = []
        chunk_output_path = input_file
        chunk_idx = 0
        print('duration less then 30')
        try:
            translation_result = translator_instance.process_audio_chunk(chunk_output_path,
                                                                         target_lang,
                                                                         chunk_idx, output_path, translation_method)
        except Exception as e:
            print(f"{e}")
            return "An Error occurred!"

        #translated_text.append(translated_text)    
        Translation_chunk_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_Translation_chunk1.wav")

        subprocess.run(['ffmpeg', '-i', Translation_chunk_output_path, '-codec:a', 'libmp3lame', output_path], check=True)
        os.remove(Translation_chunk_output_path)
        
        return translation_result, output_path

# Function to split audio into a chunk using ffmpeg
def split_audio_chunk(input_path, output_path, start_time, end_time):
    ffmpeg_cmd = f'ffmpeg -i "{input_path}" -ss {start_time} -to {end_time} -c copy "{output_path}"'
    subprocess.call(ffmpeg_cmd, shell=True)

# Function to get the duration of an audio file
def get_audio_duration(file_path):
    audio_info = librosa.get_duration(filename=file_path)
    duration_seconds = audio_info
    return duration_seconds

# Function to merge audio files
def merge_audio_files(input_files, output_file):
    merged_audio = AudioSegment.silent(duration=0)
    for input_file in input_files:
        try:
            chunk_audio = AudioSegment.from_file(input_file, format="wav")
            merged_audio += chunk_audio
        except FileNotFoundError as e:
            print(f"Error merging audio file {input_file}: {e}")
        except Exception as e:
            print(f"Error merging audio file {input_file}: {e}")
    merged_audio.export(output_file, format="wav")

# Function to delete chunk files
def delete_chunk_files(files):
    for file in files:
        try:
            os.remove(file)
        except FileNotFoundError as e:
            print(f"Error deleting file {file}: {e}")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")

def upload_audio(audio_file):
    return audio_file

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Open Translator")

    with gr.Row():
        with gr.Column():
            #gr.Markdown("## Select Translation Method:")
            translation_method = gr.Dropdown(choices=TextTranslationOption, value=TextTranslationOption[0], label="Translation Method")

            gr.Markdown("## Select Audio File:")
            audio_file = gr.File(type="filepath", label="Upload Audio File")
            audio_player = gr.Audio(label="Audio Player", interactive=True)          

            #file_title = gr.Textbox(label="Selected File Title")
            audio_file.upload(upload_file, audio_file)
            audio_file.change(upload_audio, audio_file, audio_player)

            gr.Markdown("## Select Target Language:")
            target_lang = gr.Dropdown(choices=language_choices, value="ar", label="Target Language")
            #print(target_lang)
            translate_button = gr.Button("translate")

        with gr.Column():
            translated_text = gr.Textbox(label="Translated text", lines=20, interactive=False)
            audio_output = gr.Audio(label="Translated Audio")
            translate_button.click(run_translation, inputs=[translation_method, target_lang], outputs=[translated_text, audio_output])

demo.launch(server_name="127.0.0.2", server_port=7861)