# Open Translator - Speech To Speech and Speech to text Translator - Online mode (No api keys are needed )

Open Translator, Speech To Speech Translator with voice cloning and other cool features.

## Features

1. Multi-Language Support

   - The translator supports various target languages, including English, Spanish, French, German, Japanese, Korean, Turkish, Arabic, Russian, Hebrew, Hindi, Italian, and Portuguese.

2. YouTube Downloader

   - Integrated YouTube downloader for downloading audio from YouTube videos. Users can choose between MP3 and MP4 formats.

3. Audio Conversion Tools

   - Convert Audio to MP3: Converts audio files to the MP3 format.
   - Extract Audio from Video: Extracts audio from video files and saves it as an MP3 file.

4. Memory Management Improvements

   - Improved memory management for better performance, especially with large audio files.
   - Added support for translating large audio files without running out of memory.

5. FFMpeg Integration

   - FFMpeg is utilized to split large audio files, preventing out-of-memory errors and enhancing translation quality.

6. GUI Improvements

   - Updated GUI style for a more user-friendly experience.
   - Stop button to halt the playback of translated audio files.

## Requirements

Make sure you have the following dependencies installed:

- Python >= 3.10
- Pip (Python package installer)
- [FFmpeg](https://ffmpeg.org/download.html) #Should be installed manually
- [XTTS_V2](https://huggingface.co/coqui/XTTS-v2)

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

```bash
pip install -r requirements.txt
```

5- Run the Script:

```bash
python OpenTranslator.py
```

## Options

1- File Menu:

- Convert Audio file to MP3
- Extract audio from Video
- YouTube Downloader
- Replace Audio in Video
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
[ffmpeg](https://ffmpeg.org/)
