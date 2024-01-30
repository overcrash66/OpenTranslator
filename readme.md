# Open Translator - Speech To Speech and Speech to text Translator - Online mode (No api keys are needed ) Or Local only mode

Open Translator, Speech To Speech Translator with voice cloning and other cool features.

## Features

- 16 Languages Supported:

   - The translator supports various target languages, including English, Spanish, French, German, Japanese, Korean, Turkish, Arabic, Russian, Hebrew, Hindi, Italian, Portuguese, Chinese, Czech and Ukrainian.

## Options

1- File Menu:

- Convert Audio file to MP3
- Extract audio from Video
- YouTube Downloader
- Replace Audio in Video
- Video Text Adder
- PyTranscriber (shortcut)
- Exit

2- Help Menu:

- About

3- Select Audio File:

- Browse to choose the input audio file.

4- Select Target Language:

- Choose the target language from the dropdown menu.

5- Translate:

- Click the "Translate" button to start the translation process.

6- Stop Playing Translated File:

- Click the "Stop Playing Translated File" button to stop audio playback.

## Requirements

Make sure you have the following dependencies installed:

- Python >= 3.10
- Pip (Python package installer)
- [FFmpeg](https://ffmpeg.org/download.html) #Should be installed manually

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

If you have CUDA 118 and you would like to use GPU:

```bash
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

OR use CPU:

```bash
pip install -r requirements.txt
```

5- Run the Script:

```bash
python OpenTranslator.py
```

## GUI Preview

![Redesigned (Custom)](Screenshot2.png)

## Configuration

- You can customize the translation model and other settings by modifying the script.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to:
[XTTS_V2](https://huggingface.co/coqui/XTTS-v2)
[whisper](https://github.com/openai/whisper)
[mBART-50](https://huggingface.co/SnypzZz/Llama2-13b-Language-translate)
