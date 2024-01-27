import tkinter as tk
from tkinter import Label, Button, filedialog, StringVar, OptionMenu, messagebox, ttk, DoubleVar, Menu, Entry, Frame, simpledialog, font
import threading

from transformers import WhisperProcessor, WhisperForConditionalGeneration

import logging
import os
import requests
import math
from pydub import AudioSegment
from pytube import YouTube
from pydub.utils import mediainfo
import subprocess
import time
import customtkinter
from CTkMenuBar import *
import re

import webbrowser
from .youtube_downloader import YouTubeDownloader
from .ReplaceVideoAudio import AudioReplacerGUI
from .VideoTextAdder import VideoTextAdder
from .audio_translator import CustomTranslator

from .sentence_translator import SentenceTranslator

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
		filedropdown.add_option(option="Video Text Adder", command=VideoTextAdder)
		filedropdown.add_option(option="PyTranscriber", command=self.PyTranscriber)
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
		
		self.label_target_TextTranslationOption = customtkinter.CTkLabel(pack_frame, text="Select text Translation Method:", font=("Arial", 12, "bold"),text_color="green")
		self.label_target_TextTranslationOption.pack(pady=5)

		TextTranslationOption = ["Local", "Online"]
		
		self.stringvarTextTranslationOption = customtkinter.StringVar()
		self.target_TextTranslationOption_dropdown = customtkinter.CTkOptionMenu(pack_frame, variable=self.stringvarTextTranslationOption,values=TextTranslationOption)
		self.target_TextTranslationOption_dropdown.pack(pady=5)
		self.target_TextTranslationOption_dropdown.set(TextTranslationOption[0])

		self.audio_path = ""
		
		self.label_input = customtkinter.CTkLabel(pack_frame, text="Select Audio File:", font=("Arial", 12, "bold"),text_color="green")
		self.label_input.pack(pady=5)

		self.browse_button = customtkinter.CTkButton(pack_frame, text="Browse", command=self.browse)
		self.browse_button.pack(pady=5)

		#bug if title is too long this breck the UI and change frame box size
		self.label_file_title = customtkinter.CTkLabel(pack_frame, text="Selected File Title:", font=("Arial", 12, "bold"),text_color="white", wraplength=50)
		self.label_file_title.pack(pady=5)

		self.label_target_language = customtkinter.CTkLabel(pack_frame, text="Select Target Language:", font=("Arial", 12, "bold"),text_color="green")
		self.label_target_language.pack(pady=5)

		languages = ["en", "es", "fr", "de", "ja", "ko", "tr", "ar", "ru", "he", "hi", "it", "pt"]
		
		self.translator_instance = CustomTranslator()
		
		self.stringvarlanguage = customtkinter.StringVar()
		self.target_language_dropdown = customtkinter.CTkOptionMenu(pack_frame, variable=self.stringvarlanguage,values=languages)
		self.target_language_dropdown.pack(pady=5)
		self.target_language_dropdown.set(languages[7])

		self.translate_button = customtkinter.CTkButton(pack_frame, text="Translate", command=self.translate)
		self.translate_button.pack(pady=5)

		self.stop_button = customtkinter.CTkButton(pack_frame, text="Stop Playing Translated File",command=self.stop_playing)
		self.stop_button.pack(pady=5)
		
		# Create a frame for widgets using grid
		grid_frame = Frame(master, bg="#222121")
		grid_frame.pack(side="right", padx=4)
		
		self.label_translated_text = customtkinter.CTkLabel(grid_frame, text="Translated Text:", font=("Arial", 16, "bold"), text_color="white")
		self.label_translated_text.grid(row=5, column=0, columnspan=2, pady=10)
		
		self.clear_button = customtkinter.CTkButton(grid_frame, text="Clear", command=self.clear_text)
		self.clear_button.grid(row=6, column=0, columnspan=1, pady=10)
		
		self.text_translated = tk.Text(grid_frame, height=18, width=45, wrap = 'word')
		self.text_translated.grid(row=7, column=0, columnspan=1, pady=10)

		self.save_button = customtkinter.CTkButton(grid_frame, text="Save Text Translation", command=self.save_translation)
		self.save_button.grid(row=9, column=0, columnspan=1, pady=10)
		
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
	def PyTranscriber(self):
		pytranscriber_path = r'C:\Program Files (x86)\pyTranscriber\pyTranscriber.exe'
		# Check if pyTranscriber exists
		if os.path.exists(pytranscriber_path):
			subprocess.run([pytranscriber_path])
		else:
			# Show message box to install pyTranscriber
			messagebox.showinfo("PyTranscriber Not Found",
			"Please install pyTranscriber from: https://pytranscriber.github.io/download/")	
			webbrowser.open_new(r'https://pytranscriber.github.io/download/')

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
			file_title = file_title[:50] + "..." if len(file_title) > 50 else file_title
			self.label_file_title.configure(text=f"Selected File Title: {file_title}")
	
	def clear_text(self):
		# Clear the text in the text widget
		self.text_translated.configure(state='normal')
		self.text_translated.delete("1.0", "end")
		self.text_translated.configure(state='disabled')
	
	def run_translation(self, output_path):
		input_file = self.audio_path

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
			chunk_output_path = f"{output_path}_chunk{chunk_idx + 1}.wav"

			# Update progress variable
			current_progress = (chunk_idx + 1) / num_chunks * 100
			current_progress = current_progress - 20
			current_progress = "{:.0f}".format(current_progress)

			# Update label text
			self.label_status.configure(
				text=f"Translation in progress... {current_progress}%",
				font=("Arial", 16, "bold"), text_color="red"
			)

			# Split the audio file into a chunk
			self.split_audio_chunk(self.audio_path, chunk_output_path, start_time, end_time)
			try:
				translation_result = self.translator_instance.process_audio_chunk(chunk_output_path,
															 self.target_language_dropdown.get(),
															 chunk_idx, output_path,self.target_TextTranslationOption_dropdown.get())											 
			except Exception as e:
				print(f"{e}")
				self.progress_bar.stop()
				self.label_status.configure(text="An Error occurred!",font=("Arial", 16, "bold"),text_color="red")
				
			chunk_files.append(chunk_output_path)
			
			self.text_translated.configure(state='normal')
			self.text_translated.insert('end', f"{translation_result}\n\n")
			self.text_translated.configure(state='disabled')

			Translation_chunk_output_path = f"{output_path}_Translation_chunk{chunk_idx + 1}.wav"
			Translation_chunk_files.append(Translation_chunk_output_path)
			
		# Merge individual chunk files into the final output file
		final_output_path = f"{output_path}-temp.wav"
		self.merge_audio_files(Translation_chunk_files, final_output_path)
		
		subprocess.run(['ffmpeg', '-i', final_output_path, '-codec:a', 'libmp3lame', output_path], check=True)
		os.remove(final_output_path)

		# Play the final merged audio file
		self.translator_instance.play_audio(output_path)

		# Cleanup: Delete individual chunk files
		self.delete_chunk_files(chunk_files)
		self.delete_chunk_files(Translation_chunk_files)
		chunk_files = []  # List to store individual chunk files
		Translation_chunk_files = []
		self.progress_bar.stop()

		self.label_status.configure(text="Translation complete!",font=("Arial", 16, "bold"),text_color="green")

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
				chunk_audio = AudioSegment.from_file(input_file, format="wav")

				# Append the chunk audio to the merged audio
				merged_audio += chunk_audio
			except FileNotFoundError as e:
				logging.warning(f"Error merging audio file {input_file}: {e}")
			except Exception as e:
				logging.error(f"Error merging audio file {input_file}: {e}")

		# Export the merged audio to the final output file
		try:
			merged_audio.export(output_file, format="wav")
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
				try:
					with open(output_path, "w", encoding="utf-8") as file:
						file.write(translation_text)
					print(f"Translation saved to: {output_path}")
				except Exception as e:
					print(f"Error saving translation to file: {e}")

			else:
				print("Save operation cancelled.")