from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import logging
import librosa
import torch
import sounddevice as sd
from OpenTranslator.sentence_translator import SentenceTranslator
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from TTS.api import TTS
import time
from autosub.onlineTranslator import Ctr_Autosub
from gtts import gTTS
import os
import unicodedata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTranslator:
    def __init__(self, output_dir="output"):
        self.target_language = ""
        self.source_language = ""
        self.translation_method = ""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Initialize other attributes as needed

    def load_models(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device)
        # self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def process_audio_chunk(self, input_path, target_language, src_language, chunk_idx, output_path, translation_method):
        try:
            if translation_method == 'Online' or translation_method == 'Hybrid':
                print('Starting Online translation!')
                src_language = src_language  # Assuming src_lang is defined elsewhere
                transcripts = Ctr_Autosub.generate_subtitles(source_path=input_path, output=output_path, src_language=src_language)

                print('Transcripts: ' + str(transcripts))
                translator = SentenceTranslator(src=src_language, dst=target_language)
                translated = []
                for byte_string in transcripts:
                    if byte_string is not None:
                        translated_text = translator(byte_string)
                        translated.append(translated_text)

                translated_text = ' '.join(translated)
                print('Translated text: ' + str(translated_text))

                Translation_chunk_output_path = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_Translation_chunk{chunk_idx + 1}.wav")
                print(str(Translation_chunk_output_path))
                if translation_method == 'Hybrid':
                    self.generate_audio(translated_text, Translation_chunk_output_path, target_language, input_path)
                if translation_method == 'Online':
                    tts = gTTS(translated_text, lang=target_language, slow=False)
                    tts.save(Translation_chunk_output_path)
                return translated_text

            if translation_method == 'Local':
                self.load_models()
                start_time = time.time()
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

                del input_waveform, input_sampling_rate

                end_time = time.time()
                execution_time = (end_time - start_time) / 60
                print(f"Transcription Execution time: {execution_time:.2f} minutes")

                # Fix a bug: Text validation check if we have duplicate successive words
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
                print('Speech recognition and translate to English text: ' + str(transcription))

                Translation_chunk_output_path = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_Translation_chunk{chunk_idx + 1}.wav")

                # If target language is English, skip text translation
                if target_language != 'en':
                    # Local text translation
                    print("Local text translation started..")
                    start_time = time.time()
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
                        "pt": "pt_XX",
                        "zh": "zh_CN",
                        "cs": "cs_CZ",
                        "nl": "nl_XX",
                        "pl": "pl_PL",
                    }

                    # Set the target language based on the mapping
                    model_target_language = language_mapping.get(target_language, "en_XX")

                    # Generate tokens on the GPU
                    generated_tokens = tt.generate(input_ids=input_ids, forced_bos_token_id=tokenizer.lang_code_to_id[model_target_language])

                    # Decode and join the translated text
                    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    translated_text = ", ".join(translated_text)

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
        print("Generate audio")

        # Text to speech to a file
        start_time = time.time()
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.tts.tts_to_file(text=text, speaker_wav=input_path, language=target_language, file_path=output_path)
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        print(f"Generate_audio Execution time: {execution_time:.2f} minutes")
