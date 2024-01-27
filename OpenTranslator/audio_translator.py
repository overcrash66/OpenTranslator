from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import logging
import os
from pydub import AudioSegment
import librosa
import torch
import customtkinter
from CTkMenuBar import *
from tkinter import StringVar
import sounddevice as sd
from .sentence_translator import SentenceTranslator
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import sentencepiece as spm
from TTS.api import TTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTranslator:
    def __init__(self):
        self.target_language = StringVar()
        self.target_language.set("ar")  # Default target language
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device)
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
 
    def process_audio_chunk(self, input_path, target_language, chunk_idx, output_path, Target_Text_Translation_Option):
        try:
            # Load input audio file using librosa
            input_waveform, input_sampling_rate = librosa.load(input_path, sr=None, mono=True)

            # Convert NumPy array to PyTorch tensor if needed
            if not isinstance(input_waveform, torch.Tensor):
                input_waveform = torch.tensor(input_waveform)   

            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=target_language, task="translate")

            # Ensure the input audio has a proper frame rate
            if input_sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=input_sampling_rate, new_freq=16000)
                input_waveform = resampler(input_waveform)

            # Process the input audio with the processor
            input_features = self.processor(input_waveform.numpy(), sampling_rate=16000, return_tensors="pt")

            # Move input features to the device used by the model
            input_features = {k: v.to(device) for k, v in input_features.items()}

            # Generate token ids
            predicted_ids = self.model.generate(input_features["input_features"], forced_decoder_ids=forced_decoder_ids)

            # Decode token ids to text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
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
            
            Translation_chunk_output_path = f"{output_path}_Translation_chunk{chunk_idx + 1}.wav"
            
            # Use SpeechRecognizer for translation (modify as needed)
            if target_language != "en" and Target_Text_Translation_Option == 'Online':
                print("Online text Translation started ..")
                translator = SentenceTranslator(src="en", dst=target_language)
                translated_text = translator(transcription)
                
                # Generate final audio output from translated text
                self.generate_audio(translated_text, Translation_chunk_output_path, target_language, input_path)

                logging.info(f"Processing successful. Translated text: {translated_text}")
                return translated_text
            
            if Target_Text_Translation_Option == 'Local':
                print("Local text Translation started ..")
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
                "he": "he_IL",
                "hi": "hi_IN",
                "it": "it_IT",
                "pt": "pt_PT",
                }

                # Set the target language based on the mapping
                model_Target_language = language_mapping.get(target_language, "en_XX")       
                
                # Generate tokens on the GPU
                #generated_tokens = tt.generate(**model_inputs,forced_bos_token_id=tokenizer.lang_code_to_id[model_Target_language])
                generated_tokens = tt.generate(input_ids=input_ids, forced_bos_token_id=tokenizer.lang_code_to_id[model_Target_language])
                
                # Decode and join the translated text
                translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                translated_text = ", ".join(translated_text)
                
                # Generate final audio output from translated text
                self.generate_audio(translated_text, Translation_chunk_output_path, target_language, input_path)
                
                logging.info(f"Processing successful. Translated text: {translated_text}")
                return translated_text

            else:    
                self.generate_audio(transcription, Translation_chunk_output_path, target_language, input_path)        
                logging.info(f"Processing successful. Translated text: {transcription}")
                return transcription

            # Log success
            logging.info(f"Translation successful for {input_path}. Translated text: {transcription}")

        except Exception as e:
            # Log errors
            logging.error(f"Error processing audio: {e}")
            raise  # Re-raise the exception  
        
    def generate_audio(self, text, output_path, target_language, input_path):   
        print("Generate audio")
        # Text to speech to a file
        self.tts.tts_to_file(text=text, speaker_wav=input_path, language=target_language, file_path=output_path)

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
