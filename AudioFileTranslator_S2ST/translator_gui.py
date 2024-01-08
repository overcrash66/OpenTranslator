from tkinter import Label, Button, filedialog, StringVar, OptionMenu, messagebox, ttk, DoubleVar, Menu, Entry, Frame, simpledialog
import threading
from PIL import Image, ImageTk
import pygame
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS
import torchaudio
import logging
import os
import requests
import math
from pydub import AudioSegment
from pytube import YouTube
from pydub.utils import mediainfo
import subprocess
import time
import librosa
import torch
import customtkinter
import httpx
from CTkMenuBar import *
import re
import shutil
from .youtube_downloader import YouTubeDownloader
from .ReplaceVideoAudio import AudioReplacerGUI
from .audio_translator import CustomTranslator

from .sentence_translator import SentenceTranslator
import se_extractor
from .api import BaseSpeakerTTS, ToneColorConverter

ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device = 'cpu'
#voice cloning is not accurate!!
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

customtkinter.set_appearance_mode("System")	   # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class TranslatorGUI:
	def __init__(self, master):
		self.menubar = CTkMenuBar(master=master)
		self.file = self.menubar.add_cascade("File")
		self.help = self.menubar.add_cascade("Help")

		filedropdown = CustomDropdownMenu(widget=self.file, width=100)
		filedropdown.add_option(option="Convert Audio file to MP3", command=self.Convert_Audio_Files)
		filedropdown.add_option(option="Extract audio from Video", command=self.extract_audio)
		filedropdown.add_option(option="Youtube Downloader", command=YouTubeDownloader)
		filedropdown.add_option(option="Replace Audio in Video", command=AudioReplacerGUI)
		filedropdown.add_option(option="Exit", command=master.destroy)

		helpdropdown = CustomDropdownMenu(widget=self.help, width=50)
		helpdropdown.add_option(option="About", command=self.show_about)
		master.title("Open Translator")
		master.geometry("600x640")
		master.minsize(600,640)
		master.maxsize(740,640)
		master.attributes('-fullscreen', False)

		self.label = customtkinter.CTkLabel(master=master, text="Open Translator", font=("Arial", 28, "bold"),text_color="white")
		self.label.pack(side="top", pady=10)

		# Create a frame for widgets using pack
		pack_frame = Frame(master, bg="#222121")
		pack_frame.pack(side="left", padx=2)
		
		self.audio_path = ""
		
		self.label_input = customtkinter.CTkLabel(pack_frame, text="Select Audio File:", font=("Arial", 12, "bold"),text_color="green")
		self.label_input.pack(pady=5)

		self.browse_button = customtkinter.CTkButton(pack_frame, text="Browse", command=self.browse)
		self.browse_button.pack(pady=5)

		self.label_file_title = customtkinter.CTkLabel(pack_frame, text="Selected File Title:", font=("Arial", 12, "bold"),text_color="white")
		self.label_file_title.pack(pady=5)

		self.label_target_language = customtkinter.CTkLabel(pack_frame, text="Select Target Language:", font=("Arial", 12, "bold"),text_color="green")
		self.label_target_language.pack(pady=5)

		languages = ["en", "es", "fr", "de", "ja", "ko", "tr", "ar", "ru", "he", "hi", "it", "pt"]
		
		self.translator_instance = CustomTranslator()
		
		self.stringvarlanguage = customtkinter.StringVar()
		self.target_language_dropdown = customtkinter.CTkOptionMenu(pack_frame, variable=self.stringvarlanguage,values=languages)
		self.target_language_dropdown.pack(pady=5)
		self.target_language_dropdown.set(languages[0])

		self.translate_button = customtkinter.CTkButton(pack_frame, text="Translate", command=self.translate)
		self.translate_button.pack(pady=5)

		self.stop_button = customtkinter.CTkButton(pack_frame, text="Stop Playing Translated File",command=self.stop_playing)
		self.stop_button.pack(pady=5)
		
		# Create a frame for widgets using grid
		grid_frame = Frame(master, bg="#222121")
		grid_frame.pack(side="right", padx=2)
		
		self.label_translated_text = customtkinter.CTkLabel(grid_frame, text="Translated Text:", font=("Arial", 16, "bold"), text_color="white")
		self.label_translated_text.grid(row=5, column=0, columnspan=2, pady=10)
		
		self.clear_button = customtkinter.CTkButton(grid_frame, text="Clear", command=self.clear_text)
		self.clear_button.grid(row=6, column=1, columnspan=2, pady=10)
		
		self.text_translated = customtkinter.CTkTextbox(grid_frame, height=200, width=400, wrap = 'word')
		self.text_translated.grid(row=7, column=0, columnspan=2, pady=10)
		
		self.save_button = customtkinter.CTkButton(grid_frame, text="Save Text Translation", command=self.save_translation)
		self.save_button.grid(row=9, column=0, columnspan=2, pady=10)
		
		self.progress_bar = customtkinter.CTkProgressBar(grid_frame, variable=DoubleVar(), mode='indeterminate')
		self.progress_bar.grid(row=10, column=0, columnspan=2, pady=10)

		self.label_status = customtkinter.CTkLabel(grid_frame, text="")
		self.label_status.grid(row=11, column=0, columnspan=2, pady=5)

	def translate(self):
		if self.audio_path:
			output_path = filedialog.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 Files", "*.mp3")])
			if output_path:
				translation_thread = threading.Thread(target=self.run_translation, args=(output_path,))
				translation_thread.start()
				self.progress_bar.start()
				self.label_status.configure(text="Translation in progress...",font=("Arial", 16, "bold"),text_color="red")	 

	def extract_audio(self):
		input_video = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
		if input_video != '':
			input_video_file = input_video.split("/")[-1]
			input_video_file = str(input_video_file).replace('.mp4','')
			output_audio = f"{input_video_file}.mp3"
			
			command = [
				'ffmpeg',
				'-i', input_video,
				'-vn',	# Disable video recording
				'-ac', '2',	 # Set the number of audio channels to 2
				'-ar', '44100',	 # Set the audio sample rate to 44100 Hz
				'-ab', '192k',	# Set the audio bitrate to 192 kbps
				'-f', 'mp3',  # Set the output format to mp3
				output_audio
			]

			# Run the command
			subprocess.run(command)
			
			print(f"Conversion successful: {output_audio}")
			messagebox.showinfo("Info", f"Conversion successful: {output_audio}")
		
	def Convert_Audio_Files(self):
		def is_mp3(file_path):
			try:
				result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=format_name', '-of', 'default=noprint_wrappers=1:nokey=1', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
				return result.stdout.strip() == 'mp3'
			except Exception as e:
				print(f"Error checking file format: {e}")
				messagebox.showinfo("Error", f"Error checking file format: {e}")
				return False

		def convert_to_mp3(input_file, output_file):
			try:
				subprocess.run(['ffmpeg', '-i', input_file, '-codec:a', 'libmp3lame', output_file], check=True)
				print(f"Conversion successful: {output_file}")
				messagebox.showinfo("Info", f"Conversion successful: {output_file}")
			except subprocess.CalledProcessError as e:
				print(f"Error converting to MP3: {e}")
				messagebox.showinfo("Error", f"Error converting to MP3: {e}")

		def Start(Input_file_path):
			input_file = Input_file_path
			file_title = Input_file_path.split("/")[-1]
			output_file = f"{file_title}-Converted.mp3"

			if not is_mp3(input_file):
				print(f"The input file is not a valid MP3. Converting to MP3...")
				convert_to_mp3(input_file, output_file)
			else:
				print("The input file is already an MP3.")
				messagebox.showinfo("Error", "The input file is already an MP3.")
		
		Input_file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.*")])
		
		if Input_file_path != '':
			Start(Input_file_path)
	
	def show_about(self):
		messagebox.showinfo("About", "Open Translator v1.0.0\n\nCreated by Wael Sahli\n\n")
	
	def browse(self):
		file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3")])
		print(f"Selected file: {file_path}")
		self.audio_path = file_path

		file_title = file_path.split("/")[-1]
		if file_title != "":
			self.label_file_title.configure(text=f"Selected File Title: {file_title}")
	
	def clear_text(self):
		# Clear the text in the text widget
		self.text_translated.configure(state='normal')
		self.text_translated.delete("1.0", "end")
		self.text_translated.configure(state='disabled')
	
	def run_translation(self, output_path):
		try:
			input_file = self.audio_path
			
			# use open voice to clone original voive
			target_se, audio_name = se_extractor.get_se(input_file, tone_color_converter, target_dir='processed', vad=True) 

			# Get the duration of the input audio file
			ffprobe_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input_file}"'
			input_duration = float(subprocess.check_output(ffprobe_cmd, shell=True))
			
			# Set the maximum duration for each chunk (30 seconds in this case)
			max_chunk_duration = 30

			# Calculate the number of chunks required
			num_chunks = int(input_duration / max_chunk_duration)
			print("num_chunks: "+str(num_chunks))
			chunk_files = []  # List to store individual chunk files
			Translation_chunk_files = []
			
			# Split the audio file into chunks and process each chunk
			for chunk_idx in range(num_chunks):
				
				# Calculate start and end times for each chunk
				start_time = chunk_idx * max_chunk_duration
				end_time = min((chunk_idx + 1) * max_chunk_duration, input_duration)

				# Use a consistent naming pattern for chunk files
				chunk_output_path = f"{output_path}_chunk{chunk_idx + 1}.mp3"

				# Split the audio file into a chunk
				self.split_audio_chunk(self.audio_path, chunk_output_path, start_time, end_time)
				try:
					translation_result = self.translator_instance.process_audio_chunk(chunk_output_path,
																 self.target_language_dropdown.get(),
																 chunk_idx, output_path,source_se,target_se,tone_color_converter)
				except Exception as e:
					print(f"{e}")
					self.progress_bar.stop()
					self.label_status.configure(text="An Error occurred!",font=("Arial", 16, "bold"),text_color="red")
					
				chunk_files.append(chunk_output_path)

				# Update translated text in text widget	
				if self.target_language_dropdown.get() == 'ar':#still have an issue of reverse line order for arabic text
					self.text_translated.tag_config("right", justify=customtkinter.RIGHT)
					translation_result = ' '.join(reversed(translation_result.split()))
					translation_result = translation_result.replace('.', '').replace(',', '')

				#if self.target_language_dropdown.get() != 'ar':
				self.text_translated.configure(state='normal')
				self.text_translated.insert('end', f"{translation_result}\n\n")
				self.text_translated.configure(state='disabled')
				
				Translation_chunk_output_path = f"{output_path}_Translation_chunk{chunk_idx + 1}.mp3"
				Translation_chunk_files.append(Translation_chunk_output_path)
				
			# Merge individual chunk files into the final output file
			final_output_path = f"{output_path}"
			self.merge_audio_files(Translation_chunk_files, final_output_path)

			# Play the final merged audio file
			self.translator_instance.play_audio(final_output_path)

			# Cleanup: Delete individual chunk files
			self.delete_chunk_files(chunk_files)
			self.delete_chunk_files(Translation_chunk_files)

			self.progress_bar.stop()

			self.label_status.configure(text="Translation complete!",font=("Arial", 16, "bold"),text_color="green")
			self.delete_files_in_folder('processed')
			
		except Exception as e:
			logging.error(f"Error during translation: {e}")
			raise
	
	def delete_files_in_folder(self, folder_path):
		try:
			# List all files and subfolders in the folder
			for root, dirs, files in os.walk(folder_path):
				for file in files:
					file_path = os.path.join(root, file)
					os.remove(file_path)
				
				for subfolder in dirs:
					subfolder_path = os.path.join(root, subfolder)
					shutil.rmtree(subfolder_path)

			print(f"All files and subfolders in {folder_path} have been deleted.")
		except Exception as e:
			print(f"An error occurred: {e}")		

	# Function to split audio into a chunk using ffmpeg
	def split_audio_chunk(self, input_path, output_path, start_time, end_time):
		ffmpeg_cmd = f'ffmpeg -i "{input_path}" -ss {start_time} -to {end_time} -c copy "{output_path}"'
		subprocess.call(ffmpeg_cmd, shell=True)

	def get_audio_duration(self, file_path):
		audio_info = mediainfo(file_path)
		duration_ms_str = audio_info.get("duration", "0")
		duration_ms = float(duration_ms_str)
		duration_seconds = duration_ms / 1000
		return duration_seconds

	def merge_audio_files(self, input_files, output_file):
		merged_audio = AudioSegment.silent(duration=0)
		# print("Merge started")
		for input_file in input_files:
			try:
				# Load the chunk audio
				chunk_audio = AudioSegment.from_file(input_file, format="mp3")

				# Append the chunk audio to the merged audio
				merged_audio += chunk_audio
			except FileNotFoundError as e:
				logging.warning(f"Error merging audio file {input_file}: {e}")
			except Exception as e:
				logging.error(f"Error merging audio file {input_file}: {e}")

		# Export the merged audio to the final output file
		try:
			merged_audio.export(output_file, format="mp3")
		except Exception as e:
			logging.error(f"Error exporting merged audio: {e}")

	def delete_chunk_files(self, files):
		for file in files:
			try:
				os.remove(file)
			except FileNotFoundError as e:
				logging.warning(f"Error deleting file {file}: {e}")
			except Exception as e:
				logging.error(f"Error deleting file {file}: {e}")

	def stop_playing(self):
		self.translator_instance.stop_audio()
	
	def save_translation(self):
		translation_text = self.text_translated.get("1.0", "end-1c")
		if translation_text:
			output_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
			if output_path:
				if self.target_language_dropdown.get() == 'ar':
					text = translation_text
						
					with open(output_path+'_temp', "w", encoding="utf-8") as file:
						file.write(text)
					print(f"Temp Translation saved to: {output_path}+'_temp'")		
				
					with open(output_path+'_temp', 'r', encoding="utf-8") as file2:
						lines = file2.readlines()

					reversed_lines = []
					for line in lines:
						words = line.split()
						reversed_words = ' '.join(reversed(words))
						reversed_lines.append(reversed_words)
					reversed_content = '\n'.join(reversed_lines)
					
					with open(output_path, "w", encoding="utf-8") as file3:
						file3.write(reversed_content)
					print(f"Translation saved to: {output_path}")
					os.remove(output_path+'_temp')
				else:
					try:
						with open(output_path, "w", encoding="utf-8") as file:
							file.write(translation_text)
						print(f"Translation saved to: {output_path}")
					except Exception as e:
						print(f"Error saving translation to file: {e}")

			else:
				print("Save operation cancelled.")

if __name__ == "__main__":
	run_gui()