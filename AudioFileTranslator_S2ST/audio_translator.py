from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import logging
import os
import requests
from pydub import AudioSegment
import subprocess
import time
import librosa
import torch
import customtkinter
import httpx
from CTkMenuBar import *
import re
from tkinter import StringVar
import pygame
from .sentence_translator import SentenceTranslator

from TTS.api import TTS

class CustomTranslator:
    def __init__(self):
        self.processor = None
        self.model = None
        self.target_language = StringVar()
        self.target_language.set("en")  # Default target language

    def load_model(self):
        # Load the model if it hasn't been loaded
        if self.processor is None:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

        if self.model is None:
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

        # Move the model to GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')

    def unload_model(self):
        # Unload the model if it has been loaded
        if self.processor is not None:
            del self.processor
            self.processor = None

        if self.model is not None:
            # Move the model back to CPU before deleting
            if torch.cuda.is_available():
                self.model.to('cpu')
            del self.model
            self.model = None
        
    def process_audio_chunk(self, input_path, target_language, chunk_idx, output_path, tts):
        try:
            self.load_model()
            
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
            if target_language != "en":
                translator = SentenceTranslator(src="en", dst=target_language)
                translated_text = translator(transcription)
                
                # Generate final audio output from translated text
                self.generate_audio(translated_text, Translation_chunk_output_path, target_language, input_path, tts)

                logging.info(f"Processing successful. Translated text: {translated_text}")
                return translated_text
            else:
                self.generate_audio(transcription, Translation_chunk_output_path, target_language, input_path, tts)
                
                logging.info(f"Processing successful. Translated text: {transcription}")
                return transcription

            # Log success
            logging.info(f"Translation successful for {input_path}. Translated text: {transcription}")

        except Exception as e:
            # Log errors
            logging.error(f"Error processing audio: {e}")
            raise  # Re-raise the exception
        
        finally:
            # Ensure model is unloaded and memory is cleared even if an exception occurs
            self.unload_model()    
        
    def generate_audio(self, text, output_path, target_language, input_path, tts):   
        # Text to speech to a file
        tts.tts_to_file(text=text, speaker_wav=input_path, language=target_language, file_path=output_path)

    def play_audio(self, audio_path): # disabled for now
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

    def stop_audio(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        try:    
            pygame.mixer.music.stop()
        except:
            pass
