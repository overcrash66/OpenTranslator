# Open Translator: Speech To Speech and Speech to text Translator with voice cloning and other cool features
## Can be run as Online mode ( connected to internet and make api calls using Google translate - No api keys are needed ) 
## Or Fully Locally  (using local LLM's)
## Or as a Hybrid mode (mix of Online and one single Local LLM )

## Features

- Translate from and to 17 Languages :

   - The translator supports various languages, including English, Spanish, French, German, Dutch , Japanese, Korean, Turkish, Arabic, Russian, Hebrew, Hindi, Italian, Portuguese, Chinese, Czech and Hungarian.

## Options

- File Menu available options:

- Convert Audio file to MP3
- Extract audio from Video
- YouTube Downloader
- Replace Audio in Video
- Video Text Adder
- Voice Recorder
- PyTranscriber (shortcut)
- Exit

## Requirements

Make sure you have the following dependencies installed:

- Python >= 3.10
- Pip (Python package installer)
- [FFmpeg](https://ffmpeg.org/download.html) #Should be installed manually and added to sys env path

## Usage

1- Clone the repository:

```
git clone https://github.com/overcrash66/OpenTranslator.git
```

2- Navigate to folder:

```
cd OpenTranslator
```

3- Create a vitrual env:

```
py -3.10 -m venv venv
```

```
venv\Scripts\activate
```

4- Install the required Python packages using:

If you would like to use CUDA 118 - GPU:

PY -3.10
```bash
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

PY -3.12
```bash
pip install torch==2.2.1+cu118 torchaudio==2.2.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Install mecab 
https://github.com/ikegami-yukino/mecab/releases


```bash
pip install -r requirements_Py312.txt
```

OR by default you use CPU only:

```bash
pip install -r requirements.txt
```

5- Run the Script:

```bash
python OpenTranslator.py
```

Or Local mode (using a set of LLM's) for audio file translation only, using a WEB UI (Gradio)

```
python WebUI.py

```

## GUI Preview

![Redesigned (Custom)](Screenshot2.png)

![Web](Screenshot.png)

## Audio translation examples:
- [ar-test.mp3](./testResults/ar-test.mp3)
- [cn-test.mp3](./testResults/cn-test.mp3)
- [cs-test.mp3](./testResults/cs-test.mp3)
- [de-test.mp3](./testResults/de-test.mp3)
- [en-test.mp3](./testResults/en-test.mp3)
- [es-test.mp3](./testResults/es-Original-Audio.mp3)
- [fr-test.mp3](./testResults/fr-test.mp3)
- [hi-test.mp3](./testResults/hi-test.mp3)
- [hu-test.mp3](./testResults/hu-test.mp3)
- [it-test.mp3](./testResults/it-test.mp3)
- [jp-test.mp3](./testResults/jp-test.mp3)
- [ko-test.mp3](./testResults/ko-test.mp3)
- [nl-test.mp3](./testResults/nl-test.mp3)
- [pl-test.mp3](./testResults/pl-test.mp3)
- [pt-test.mp3](./testResults/pt-test.mp3)
- [ru-test.mp3](./testResults/ru-test.mp3)
- [tr-test.mp3](./testResults/tr-test.mp3)

## Configuration

- You can customize the translation models and other settings by modifying the script.

## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to:
[XTTS_V2](https://huggingface.co/coqui/XTTS-v2)
[whisper v3 Large](https://huggingface.co/openai/whisper-large-v3)
[Llama2-13b-Language-translate](https://huggingface.co/SnypzZz/Llama2-13b-Language-translate)
[autosub](https://github.com/agermanidis/autosub)
[gTTS](https://github.com/pndurette/gTTS)
