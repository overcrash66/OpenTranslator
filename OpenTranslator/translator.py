from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import logging
import librosa
import torch
import sounddevice as sd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from TTS.api import TTS
import time
import os
import unicodedata

from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTranslator:
    def __init__(self, output_dir="output"):
        self.target_language = ""
        self.source_language = ""
        self.translation_method = ""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_whisper_model(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device)

    def unload_whisper_model(self):
        del self.processor
        del self.model

    def load_mbart_model(self):
        self.mbart_model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate").to(device)
        self.mbart_tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX", device=device)

    def unload_mbart_model(self):
        del self.mbart_model
        del self.mbart_tokenizer

    def load_tts_model(self):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def unload_tts_model(self):
        del self.tts

    def process_audio_chunk(self, input_path, target_language, chunk_idx, output_path, translation_method , batch_size=4):
        try:
            start_time = time.time()

            self.load_whisper_model()

            # Load audio waveform
            input_waveform, input_sampling_rate = librosa.load(input_path, sr=None, mono=True)
            
            if not isinstance(input_waveform, torch.Tensor):
                input_waveform = torch.tensor(input_waveform)

            if input_sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=input_sampling_rate, new_freq=16000)
                input_waveform = resampler(torch.tensor(input_waveform).clone().detach()).numpy()

            # Prepare forced decoder IDs
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=target_language, task="translate")

            # Create batches of input features
            input_features = self.processor(
                    input_waveform,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
            input_features = {k: v.to(device) for k, v in input_features.items()}
            input_batches = torch.split(input_features["input_features"], batch_size, dim=0)

            # Process batches
            transcriptions = []
            for batch in input_batches:
                with torch.no_grad():
                    predicted_ids = self.model.generate(batch, forced_decoder_ids=forced_decoder_ids, max_length=448)
                    transcriptions.extend(self.processor.batch_decode(predicted_ids, skip_special_tokens=True))

            # Combine transcriptions
            transcription = " ".join(transcriptions)

            del input_waveform, input_sampling_rate

            end_time = time.time()
            execution_time = (end_time - start_time) / 60
            print(f"Transcription Execution time: {execution_time:.2f} minutes")

            words = transcription.split()
            cleaned_words = [words[0]]
            for word in words[1:]:
                if word != cleaned_words[-1]:
                    cleaned_words.append(word)
            cleaned_str = ' '.join(cleaned_words)

            sentences = cleaned_str.split('.')
            cleaned_sentences = [sentences[0]]
            for sentence in sentences[1:]:
                if sentence != cleaned_sentences[-1]:
                    cleaned_sentences.append(sentence)
            cleaned_transcription = '.'.join(cleaned_sentences)

            transcription = cleaned_transcription
            print('Speech recognition and translate to English text: ' + str(transcription))

            Translation_chunk_output_path = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_Translation_chunk{chunk_idx + 1}.wav")

            if target_language != 'en' and translation_method == 'Llama2-13b':
                print("Local text translation started..")
                start_time = time.time()
                self.load_mbart_model()

                inputs = self.mbart_tokenizer(transcription, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)

                language_mapping = {
                    "en": "en_XX", "es": "es_XX", "fr": "fr_XX", "de": "de_DE",
                    "ja": "ja_XX", "ko": "ko_KR", "tr": "tr_TR", "ar": "ar_AR",
                    "ru": "ru_RU", "he": "he_IL", "hi": "hi_IN", "it": "it_IT",
                    "pt": "pt_XX", "zh": "zh_CN", "cs": "cs_CZ", "nl": "nl_XX", "pl": "pl_PL",
                }
                model_target_language = language_mapping.get(target_language, "en_XX")

                # Generate tokens on the GPU
                generated_tokens = self.mbart_model.generate(input_ids=input_ids, forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id[model_target_language])

                # Decode and join the translated text
                translated_text = self.mbart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                translated_text = ", ".join(translated_text)

                self.unload_mbart_model()
                
                print('Mbart Translation: '+ str(translated_text))
                end_time = time.time()
                execution_time = (end_time - start_time) / 60
                print(f"Transcription Execution time: {execution_time:.2f} minutes")

            if target_language == 'en':
                translated_text = transcription

            if target_language != 'en' and translation_method == 'TowerInstruct-7B':
                translated_text = self.validate_translation(transcription, target_language)
            
            self.generate_audio(translated_text, Translation_chunk_output_path, target_language, input_path)
            
            self.unload_whisper_model()
            return translated_text
            

        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            return "An Error occurred!", None

    def validate_translation(self, source_text, target_language):
        print('validate_translation started ..')
        start_time = time.time()

        languages = {
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

        code_to_language = {code: lang for lang, code in languages.items()}
        target_language = code_to_language.get(target_language, "Unknown language")
        
        #supports 10 languages: English, German, French, Spanish, Chinese, Portuguese, Italian, Russian, Korean, and Dutch
        pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-7B-v0.2", torch_dtype=torch.bfloat16, device_map=device)
        # We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "user",
                "content": (
                    f"Translate the following text from English into {target_language}.\n"
                    f"English: {source_text}\n"
                    f"{target_language}:"
                ),
            }
        ]

        #print(target_language)
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
        generated_text = outputs[0]["generated_text"]
        
        #translated_text = generated_text.split("English:")[-1].strip()

        # Further sanitize to remove undesired formatting tokens
        generated_text = (
            generated_text.replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
            .strip()
        )

        # Define the unwanted substrings in a list
        unwanted_substrings = [
            target_language,
            source_text,
            'assistant',
            'Translate the following text from English into .',
            '\n',
            'English:',
            ':'
        ]

        # Remove the unwanted substrings
        translated_text = generated_text.split("\n", 1)[-1].strip()  # Split and strip the first line
        for substring in unwanted_substrings:
            translated_text = translated_text.replace(substring, '')

        print(f'validate_translation: {translated_text}')
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        print(f"Generate_audio Execution time: {execution_time:.2f} minutes")
        return translated_text

    def generate_audio(self, text, output_path, target_language, input_path):
        print("Generate audio")
        start_time = time.time()

        self.load_tts_model()

        self.tts.tts_to_file(text=text, speaker_wav=input_path, language=target_language, file_path=output_path)
        
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        print(f"Generate_audio Execution time: {execution_time:.2f} minutes")

        self.unload_tts_model()
