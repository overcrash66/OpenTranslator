from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import logging
import librosa
import torch
from CTkMenuBar import *
from tkinter import StringVar
import sounddevice as sd
import soundfile as sf
import numpy as np

# CRITICAL: Patch torchaudio BEFORE importing TTS
# TTS/XTTS captures torchaudio.load at import time, so we must patch first
def _custom_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, *args, **kwargs):
    """Custom torchaudio.load using soundfile to avoid TorchCodec dependency"""
    start = frame_offset
    with sf.SoundFile(uri) as sf_desc:
        sr = sf_desc.samplerate
        if num_frames == -1:
            frames_to_read = sf_desc.frames - start
        else:
            frames_to_read = num_frames
        sf_desc.seek(start)
        data = sf_desc.read(frames=frames_to_read, dtype='float32')
        
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        if channels_first:
            waveform = waveform.t()
    return waveform, sr

def _custom_save(uri, src, sample_rate, channels_first=True, *args, **kwargs):
    """Custom torchaudio.save using soundfile"""
    waveform = src.detach().cpu().numpy()
    if channels_first and waveform.ndim > 1:
        waveform = waveform.T
    sf.write(uri, waveform, sample_rate)

# Apply patches to torchaudio module
torchaudio.load = _custom_load
torchaudio.save = _custom_save

# Now import TTS (it will see our patched torchaudio)
from TTS.api import TTS

# Also patch inside XTTS module directly (it may have its own import reference)
try:
    import TTS.tts.models.xtts as xtts_module
    xtts_module.torchaudio.load = _custom_load
except Exception:
    pass

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import time

from autosub.onlineTranslator import Ctr_Autosub
from autosub import Translator, GOOGLE_SPEECH_API_KEY
from gtts import gTTS

from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTranslator:
    def __init__(self):
        # Note: torchaudio patching is now done at module level (before TTS import)
        
        self.target_language = StringVar()
        self.target_language.set("ar")  # Default target language
        self.pipe = None
        # HY-MT1.5-7B model
        self.hy_model = None
        self.hy_tokenizer = None
        # HY-MT1.5-1.8B model (smaller, faster)
        self.hy_small_model = None
        self.hy_small_tokenizer = None

    def load_models(self):
        print("Loading Whisper Model via Pipeline...")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "distil-whisper/distil-large-v3"
        
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=self.torch_dtype,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                model_kwargs={"attn_implementation": "sdpa"}, # using sdpa as it was confirmed available
            )
            print("Pipeline loaded successfully.")
        except Exception as e:
             print(f"Failed to load optimized pipeline, falling back to default: {e}")
             self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=self.torch_dtype,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )

    def load_hy_model(self):
        print("Loading Tencent HY-MT1.5-7B Model...")
        model_id = "tencent/HY-MT1.5-7B"
        try:
            self.hy_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.hy_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
            print("HY-MT1.5-7B loaded successfully.")
        except Exception as e:
            print(f"Failed to load HY-MT1.5-7B: {e}")
            raise e

    def load_hy_small_model(self):
        """Load the smaller, faster HY-MT1.5-1.8B model"""
        print("Loading Tencent HY-MT1.5-1.8B Model...")
        model_id = "tencent/HY-MT1.5-1.8B"
        try:
            self.hy_small_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.hy_small_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
            print("HY-MT1.5-1.8B loaded successfully.")
        except Exception as e:
            print(f"Failed to load HY-MT1.5-1.8B: {e}")
            raise e
 
 
    def process_audio_chunk(self, input_path, target_language,src_lang, chunk_idx, output_path, Target_Text_Translation_Option, local_model_name='Llama2-13b'):
        try:
            if Target_Text_Translation_Option == 'Online' or Target_Text_Translation_Option == 'Hybrid':
                print('Starting Online translation!')
                src_language = src_lang
                transcripts = Ctr_Autosub.generate_subtitles(source_path = input_path,output = output_path,src_language = src_language)
                
                print('transcripts: '+str(transcripts))
                translator = SentenceTranslator(src=src_language, dst=target_language)
                #ERROR:root:Error processing audio: sequence item 10: expected str instance, NoneType found
                translated=[]
                for byte_string in transcripts:
                    #byte_string = str(byte_string).replace('None', '')
                     if byte_string is not None:
                        translated_text = translator(byte_string)
                        translated.append(translated_text)
                    
                translated_text = ' '.join(translated)
                print('translated_text: '+str(translated_text))
                
                Translation_chunk_output_path = f"{output_path}_Translation_chunk{chunk_idx + 1}.wav"
                print(str(Translation_chunk_output_path))
                if Target_Text_Translation_Option == 'Hybrid':
                    self.generate_audio(translated_text, Translation_chunk_output_path, target_language, input_path)   
                if Target_Text_Translation_Option == 'Online':
                    tts = gTTS(translated_text, lang=target_language, slow=False)
                    tts.save(Translation_chunk_output_path)
                return translated_text

            if Target_Text_Translation_Option == 'Local':
                if self.pipe is None:
                    self.load_models()    
                start_time = time.time()
                
                # Using the pipeline for transcription
                # generate_kwargs={"task": "transcribe", "language": target_language} if we wanted to force language logic
                # But here we are translating? Wait, the original code had forced_decoder_ids for 'translate' task.
                # The prompt says: "replace whisper for speech to text translation"
                # If target_language is provided, we might want to use it.
                # However, distil-whisper is mainly English.
                # Original code used: forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=target_language, task="translate")
                
                # Check if target language is English for translation task, or if we should just transcribe.
                # If the user wants translation to X, whisper can do X -> X (transcribe) or X -> English (translate).
                # Distil-whisper is English only for target usually? No, let's check.
                # The user seems to use it for translation?
                # "forced_decoder_ids(language=target_language)"
                
                # Pipeline usage:
                generate_kwargs = {"language": target_language, "task": "translate"}

                outputs = self.pipe(
                    input_path,
                    chunk_length_s=30,
                    batch_size=8,
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs
                )
                
                transcription = outputs["text"].strip()

                end_time = time.time()
                execution_time = (end_time - start_time) / 60
                print(f"Transcription Execution time: {execution_time:.2f} minutes")

                #fix a bug: Text Validation check if we have duplicate successive words
                words = transcription.split()
                cleaned_words = [words[0]]

                for word in words[1:]:
                    if word != cleaned_words[-1]:
                        cleaned_words.append(word)

                cleaned_str = ' '.join(cleaned_words)
                
                transcription = cleaned_str

                # Fix duplicate successive sentences
                sentences = transcription.split('.')
                cleaned_sentences = [sentences[0]]

                for sentence in sentences[1:]:
                    if sentence != cleaned_sentences[-1]:
                        cleaned_sentences.append(sentence)

                cleaned_transcription = '.'.join(cleaned_sentences)

                transcription = cleaned_transcription
                print('Speech recognition and translate to english text: '+str(transcription))
                
                Translation_chunk_output_path = f"{output_path}_Translation_chunk{chunk_idx + 1}.wav"
                
                #if target language is English skip text translation
                if target_language != 'en':
                    #Local text Translation
                    print("Local text Translation started ..")
                    start_time = time.time()
                    
                    if local_model_name == 'Llama2-13b':
                        tt = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate").to(device)
                        tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX", device=device)
                        
                        # Tokenize and convert to PyTorch tensor
                        inputs = tokenizer(transcription, return_tensors="pt")
                        input_ids = inputs["input_ids"].to(device)
    
                        # Map target languages to model language codes
                        language_mapping = {
                        "en": "en_XX",
                        "es": "es_XX",
                        "fr": "fr_XX",
                        "de": "de_DE",
                        "ja": "ja_XX",
                        "ko": "ko_KR",
                        "tr": "tr_TR",
                        "ar": "ar_AR",
                        "ru": "ru_RU",
                        "hu": "he_IL",
                        "hi": "hi_IN",
                        "it": "it_IT",
                        "pt": "pt_XX",
                        "zh": "zh_CN",
                        "cs": "cs_CZ",
                        "nl": "nl_XX",
                        "pl": "pl_PL",
                        }
    
                        # Set the target language based on the mapping
                        model_Target_language = language_mapping.get(target_language, "en_XX")       
                        
                        # Generate tokens on the GPU
                        generated_tokens = tt.generate(input_ids=input_ids, forced_bos_token_id=tokenizer.lang_code_to_id[model_Target_language])
                        
                        # Decode and join the translated text
                        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        translated_text = ", ".join(translated_text)
                    
                    elif local_model_name == 'HY-MT1.5-7B':
                        if self.hy_model is None:
                            self.load_hy_model()
                        
                        # Language code to name mapping
                        code_to_name = {
                            "en": "English", "es": "Spanish", "fr": "French", "de": "German",
                            "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "ar": "Arabic",
                            "ru": "Russian", "hu": "Hebrew", "hi": "Hindi", "it": "Italian",
                            "pt": "Portuguese", "zh": "Chinese", "cs": "Czech", "nl": "Dutch",
                            "pl": "Polish"
                        }
                        target_lang_name = code_to_name.get(target_language, "English")
                        
                        # Build prompt according to official Tencent documentation
                        if target_language == 'zh':
                            prompt_text = f"将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{transcription}"
                        else:
                            prompt_text = f"Translate the following segment into {target_lang_name}, without additional explanation.\n\n{transcription}"
                        
                        # Use chat template as required by HY-MT1.5-7B (per official documentation)
                        messages = [
                            {"role": "user", "content": prompt_text}
                        ]
                        
                        # Apply chat template with add_generation_prompt=False as per docs
                        tokenized_chat = self.hy_tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=False,
                            return_tensors="pt"
                        ).to(device)
                        
                        input_length = tokenized_chat.shape[1]
                        
                        # Use recommended generation parameters from Tencent documentation
                        # top_k=20, top_p=0.6, repetition_penalty=1.05, temperature=0.7
                        generated_ids = self.hy_model.generate(
                            tokenized_chat,
                            max_new_tokens=512,
                            do_sample=True,
                            top_k=20,
                            top_p=0.6,
                            temperature=0.7,
                            repetition_penalty=1.05,
                            eos_token_id=self.hy_tokenizer.eos_token_id,
                            pad_token_id=self.hy_tokenizer.pad_token_id,
                        )
                        
                        # Extract only the newly generated tokens (exclude input tokens)
                        new_tokens = generated_ids[:, input_length:]
                        translated_text = self.hy_tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
                        
                        # Also decode full output for debugging
                        full_output = self.hy_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        print(f"HY-MT1.5-7B Raw Output: {full_output}")
                        print(f"HY-MT1.5-7B Extracted Translation: {translated_text}")

                    elif local_model_name == 'HY-MT1.5-1.8B':
                        # Smaller, faster model - same logic as 7B
                        if self.hy_small_model is None:
                            self.load_hy_small_model()
                        
                        # Language code to name mapping
                        code_to_name = {
                            "en": "English", "es": "Spanish", "fr": "French", "de": "German",
                            "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "ar": "Arabic",
                            "ru": "Russian", "hu": "Hebrew", "hi": "Hindi", "it": "Italian",
                            "pt": "Portuguese", "zh": "Chinese", "cs": "Czech", "nl": "Dutch",
                            "pl": "Polish"
                        }
                        target_lang_name = code_to_name.get(target_language, "English")
                        
                        # Build prompt according to official Tencent documentation
                        if target_language == 'zh':
                            prompt_text = f"将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{transcription}"
                        else:
                            prompt_text = f"Translate the following segment into {target_lang_name}, without additional explanation.\n\n{transcription}"
                        
                        # Use chat template as required by HY-MT models
                        messages = [
                            {"role": "user", "content": prompt_text}
                        ]
                        
                        # Apply chat template
                        tokenized_chat = self.hy_small_tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=False,
                            return_tensors="pt"
                        ).to(device)
                        
                        input_length = tokenized_chat.shape[1]
                        
                        # Use recommended generation parameters from Tencent documentation
                        generated_ids = self.hy_small_model.generate(
                            tokenized_chat,
                            max_new_tokens=512,
                            do_sample=True,
                            top_k=20,
                            top_p=0.6,
                            temperature=0.7,
                            repetition_penalty=1.05,
                            eos_token_id=self.hy_small_tokenizer.eos_token_id,
                            pad_token_id=self.hy_small_tokenizer.pad_token_id,
                        )
                        
                        # Extract only the newly generated tokens
                        new_tokens = generated_ids[:, input_length:]
                        translated_text = self.hy_small_tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
                        
                        # Debugging output
                        full_output = self.hy_small_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        print(f"HY-MT1.5-1.8B Raw Output: {full_output}")
                        print(f"HY-MT1.5-1.8B Extracted Translation: {translated_text}")

                    logging.info(f"Processing successful. Translated text: {translated_text}")
                    end_time = time.time()
                    execution_time = (end_time - start_time) / 60
                    print(f"Local Translation Execution time: {execution_time:.2f} minutes")

                if target_language == 'en':
                    translated_text = transcription
                            
                # Generate final audio output from translated text
                self.generate_audio(translated_text, Translation_chunk_output_path, target_language, input_path)    
                
                # Log success
                logging.info(f"Translation successful for {input_path}. Translated text: {transcription}")
                return translated_text
        
        except Exception as e:
            # Log errors
            logging.error(f"Error processing audio: {e}")
            raise  # Re-raise the exception  
       
    def generate_audio(self, text, output_path, target_language, input_path):   
        """Generate audio using Edge-TTS (Microsoft Neural Voices)"""
        import asyncio
        import edge_tts
        
        print("Generate audio using Edge-TTS")
        print(f"  Text: {text[:100]}..." if len(text) > 100 else f"  Text: {text}")
        print(f"  Target language: {target_language}")
        
        start_time = time.time()
        
        # Edge-TTS voice mapping - high quality neural voices for each language
        edge_tts_voices = {
            "en": "en-US-AriaNeural",
            "es": "es-ES-ElviraNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
            "it": "it-IT-ElsaNeural",
            "pt": "pt-BR-FranciscaNeural",
            "pl": "pl-PL-ZofiaNeural",
            "tr": "tr-TR-EmelNeural",
            "ru": "ru-RU-SvetlanaNeural",
            "nl": "nl-NL-ColetteNeural",
            "cs": "cs-CZ-VlastaNeural",
            "ar": "ar-SA-ZariyahNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
            "hu": "hu-HU-NoemiNeural",
            "ko": "ko-KR-SunHiNeural",
            "ja": "ja-JP-NanamiNeural",
            "hi": "hi-IN-SwaraNeural",
        }
        
        voice = edge_tts_voices.get(target_language, "en-US-AriaNeural")
        print(f"  Using voice: {voice}")
        
        # Edge-TTS generates MP3 by default, convert output path if needed
        # If output_path ends with .wav, we'll generate MP3 first then convert
        mp3_output = output_path.replace('.wav', '.mp3') if output_path.endswith('.wav') else output_path
        
        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(mp3_output)
        
        # Run async function
        try:
            asyncio.run(_generate())
            print(f"  Generated: {mp3_output}")
            
            # If original output was .wav, convert MP3 to WAV for compatibility
            if output_path.endswith('.wav') and mp3_output != output_path:
                import subprocess
                import os
                # Use ffmpeg to convert MP3 to WAV
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-i', mp3_output, 
                        '-acodec', 'pcm_s16le', '-ar', '24000', 
                        output_path
                    ], check=True, capture_output=True)
                    os.remove(mp3_output)  # Clean up temp MP3
                    print(f"  Converted to: {output_path}")
                except Exception as conv_err:
                    print(f"  Note: Could not convert to WAV, using MP3: {conv_err}")
                    # Keep the MP3 file
                    import shutil
                    shutil.move(mp3_output, output_path.replace('.wav', '.mp3'))
                    
        except Exception as e:
            print(f"  Edge-TTS error: {e}")
            raise
        
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        print(f"Generate_audio Execution time: {execution_time:.2f} minutes")

    def play_audio(self, audio_path):
        self.audio_data, self.sample_rate = librosa.load(audio_path, sr=None)
        sd.play(self.audio_data, self.sample_rate)
 
    def stop_audio(self):
        try:
            sd.stop()
            del self.audio_data, self.sample_rate
        except Exception as e:
            print(str(e))
            pass    