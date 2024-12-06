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
from datetime import datetime
import shlex

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

# Define the translation options
TextTranslationOption = ["Llama2-13b","TowerInstruct-7B"]

# Function to toggle button state
def toggle_button():
    # Access the current state without parentheses
    if state.value:  # Current state is True
        state.value = False  # Toggle to False
        return "OFF"
    else:  # Current state is False
        state.value = True  # Toggle to True
        return "ON"

# Initial button state
initial_state = False
initial_label = "OFF"

# Function to handle file uploads
def upload_file(file):
    global audio_path
    audio_path = file.name

def enhance_audio(input_file, reference_file, output_file, bitrate="320k", volume_boost="10dB"):
    """
    Enhances the input audio and matches the timing of the reference audio file.
    """
    try:
        # Verify that the input file and reference file exist
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.isfile(reference_file):
            raise FileNotFoundError(f"Reference file not found: {reference_file}")

        # Extract the duration of the reference file (to match timing)
        command_duration = [
            "ffmpeg",
            "-i", reference_file,
            "-f", "null", 
            "-"
        ]
        result = subprocess.run(command_duration, stderr=subprocess.PIPE, text=True)
        duration_line = [line for line in result.stderr.splitlines() if "Duration" in line]
        if not duration_line:
            raise Exception("Unable to extract duration from reference file")
        
        duration_str = duration_line[0].split("Duration:")[1].split(",")[0].strip()
        hours, minutes, seconds = map(float, duration_str.split(":"))
        reference_duration = hours * 3600 + minutes * 60 + seconds  # duration in seconds

        # Define filters for audio processing
        noise_reduction_filter = "afftdn"  # Adaptive filter for noise reduction
        normalization_filter = "loudnorm"  # EBU R128 normalization
        dynamic_compression_filter = "acompressor"  # Dynamic range compression
        equalizer_filter = "equalizer=f=1000:t=q:w=0.5:g=5"
        volume_filter = f"volume={volume_boost}"
        echo_cancellation_filter = "aecho=0.8:0.88:6:0.4"

        # Combine the filters
        audio_filters = (
            f"{noise_reduction_filter},"
            f"{normalization_filter},"
            f"{dynamic_compression_filter},"
            f"{echo_cancellation_filter},"
            f"{equalizer_filter},"
            f"{volume_filter}"
        )

        # Build the ffmpeg command to enhance the audio
        command_enhance = [
            "ffmpeg",
            "-i", shlex.quote(input_file),
            "-af", audio_filters,
            "-b:a", bitrate,  # High bitrate for best quality
            "-async", "1",  # Ensure timing consistency
            shlex.quote(output_file)
        ]
        print(f"Running command to enhance audio: {' '.join(command_enhance)}")

        # Execute the command to enhance the audio
        subprocess.run(command_enhance, check=True)

        tempOutputFile = str(output_file)+'_tt.mp3'

        # Now, adjust the duration of the enhanced audio to match the reference file
        command_adjust_timing = [
            "ffmpeg",
            "-i", output_file,
            "-t", str(reference_duration),  # Set duration to match reference
            "-c", "copy",  # Copy the audio codec to avoid re-encoding
            tempOutputFile
        ]
        print(f"Running command to adjust timing: {' '.join(command_adjust_timing)}")

        # Execute the command to adjust the duration of the enhanced audio
        subprocess.run(command_adjust_timing, check=True)

        print(f"Enhanced audio saved to {output_file}, timing matched to reference file")

        # Replace the original file with the enhanced version
        os.remove(output_file)
        os.rename(tempOutputFile, output_file)

        print(f"Replaced original file with enhanced audio: {input_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error during audio enhancement: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Function to run the translation process
def run_translation(translation_method, target_lang):
    valid_methods = ['Llama2-13b', 'TowerInstruct-7B']
    if translation_method not in valid_methods:
        raise ValueError(f"Invalid translation method: {translation_method}")
    if translation_method == 'Llama2-13b':
        target_lang = languages.get(target_lang)
    if translation_method == 'TowerInstruct-7B':
        target_lang = TowerInstruct_languages.get(target_lang)   

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_translated_To_{target_lang}_{translation_method}_{current_time}.mp3")
    input_file = audio_path
    print(audio_path)
    input_duration = get_audio_duration(input_file)
    max_chunk_duration = 30
    num_chunks = int(input_duration / max_chunk_duration)
    print('input_duration: '+str(input_duration))

    if input_duration > 30: 
        print('Duration more then 30 sec - num_chunks: '+str(num_chunks))
        chunk_files = []
        Translation_chunk_files = []
        translated_text = []
        
        for chunk_idx in range(num_chunks):
            print('Current Chunk_idx'+str(chunk_idx))
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
        
        merge_audio_files(Translation_chunk_files, final_output_path)

        if state.value == True:
            print('Improve_Audio_Quality started ..')
            tmp_output_file = str(output_path)+'_tmp.mp3'
            #convert to mp3 final audio file
            subprocess.run(['ffmpeg', '-i', final_output_path, '-codec:a', 'libmp3lame', tmp_output_file], check=True)
            reference_file = input_file
            enhance_audio(tmp_output_file, reference_file, output_path)
            os.remove(final_output_path)
            os.remove(tmp_output_file)
        else:
            subprocess.run(['ffmpeg', '-i', final_output_path, '-codec:a', 'libmp3lame', output_path], check=True)
            os.remove(final_output_path)  

        delete_chunk_files(chunk_files)
        delete_chunk_files(Translation_chunk_files)
        chunk_files = []  # List to store individual chunk files
        Translation_chunk_files = []
        
        translation_result = ', '.join(translated_text)
        return translation_result, output_path

    if input_duration <= 30 and num_chunks <= 1:
        chunk_output_path = input_file
        
        print('duration less or equal to 30 sec')
        try:
            translation_result = translator_instance.process_audio_chunk(chunk_output_path,
                                                                         target_lang,
                                                                         chunk_idx, output_path, translation_method)
        except Exception as e:
            print(f"{e}")
            return "An Error occurred!"

        Translation_chunk_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_Translation_chunk1.wav")

        #add audio timing hack
        if state.value == True:
            tmp_output_file = str(output_path)+'_tmp.mp3'
            subprocess.run(['ffmpeg', '-i', Translation_chunk_output_path, '-codec:a', 'libmp3lame', tmp_output_file], check=True)
            reference_file = input_file
            enhance_audio(tmp_output_file, reference_file, output_path)
            os.remove(Translation_chunk_output_path)
            os.remove(tmp_output_file)

        else:
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

TowerInstruct_languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Korean": "ko",
    "Russian": "ru",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese (Mandarin)": "zh",
    "Dutch": "nl"
}

model_languages = {
    "Llama2-13b": list(languages.keys()),
    "TowerInstruct-7B": list(TowerInstruct_languages.keys())
}

def update_languages(selected_model):
    supported_languages = model_languages[selected_model]
    return gr.update(choices=supported_languages, value=supported_languages[0])

# Define the Gradio interface
with gr.Blocks() as demo:
    demo.clear()
    gr.Markdown("# Open Translator WebUi")

    with gr.Row():
        with gr.Column():
            translation_method = gr.Dropdown(choices=TextTranslationOption, value=TextTranslationOption[0], label="Translation Method") 

            gr.Markdown("## Select Audio File:")
            audio_file = gr.File(type="filepath", label="Select The Audio File")
            audio_player = gr.Audio(label="Audio Player", interactive=True)          

            audio_file.upload(upload_file, audio_file)
            audio_file.change(upload_audio, audio_file, audio_player)

            gr.Markdown("## Optimize Output Audio file Quality:")
            state = gr.State(value=initial_state)  # Internal state to track the toggle
            button = gr.Button(initial_label)

            # Set up button click behavior
            button.click(
                toggle_button,
                outputs=[button]
            )

            gr.Markdown("## Select Language:")
            target_lang = gr.Dropdown(
                choices=model_languages["Llama2-13b"], 
                value=model_languages["Llama2-13b"][0], 
                label="Translate To"
            )
            
            translation_method.change(
                update_languages, 
                inputs=translation_method, 
                outputs=target_lang
            )

            translate_button = gr.Button("Start Translation")

        with gr.Column():
            translated_text = gr.Textbox(label="Translated text Result", lines=20, interactive=False)
            audio_output = gr.Audio(label="Translated Audio Result")
            translate_button.click(run_translation, inputs=[translation_method, target_lang], outputs=[translated_text, audio_output])


demo.launch(server_name="127.0.0.1", server_port=7861)