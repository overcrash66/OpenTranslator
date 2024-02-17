import tkinter as tk
from tkinter import filedialog, messagebox, DoubleVar, Frame
import logging
import os
from pydub import AudioSegment
from pydub.utils import mediainfo
import subprocess
import customtkinter
from CTkMessagebox import CTkMessagebox
from CTkMenuBar import *
import threading
import webbrowser
from .audio_translator import CustomTranslator
import webbrowser
import ctypes
ctypes.windll.user32.SetProcessDPIAware()

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
		filedropdown.add_option(option="Youtube Downloader", command=self.YouTubeDownloader)
		filedropdown.add_option(option="Replace Audio in Video", command=self.AudioReplacerGUI)
		filedropdown.add_option(option="Video Text Adder", command=self.VideoTextAdder)
		filedropdown.add_option(option="PyTranscriber", command=self.PyTranscriber)
		filedropdown.add_option(option="Exit", command=master.destroy)

		helpdropdown = CustomDropdownMenu(widget=self.help, width=50)
		helpdropdown.add_option(option="About", command=self.show_about)
		master.title("Open Translator")
		master.geometry("740x680")
		master.minsize(740,680)
		master.maxsize(740,680)
		master.attributes('-fullscreen', False)

		self.label = customtkinter.CTkLabel(master=master, text="Open Translator", font=("Comic Sans MS", 30, "bold"),text_color="red")
		self.label.pack(side="top", pady=10)

		# Create a frame for widgets using pack
		pack_frame = Frame(master, bg="#222121")
		pack_frame.pack(side="left", padx=20 ,pady=20)
		
		self.label_target_TextTranslationOption = customtkinter.CTkLabel(pack_frame, text="Select Translation Method:", font=("Arial", 12, "bold"),text_color="green")
		self.label_target_TextTranslationOption.pack(pady=5)
		TextTranslationOption = ["Local", "Online", "Hybrid"]	
		self.stringvarTextTranslationOption = customtkinter.StringVar()
		self.target_TextTranslationOption_dropdown = customtkinter.CTkOptionMenu(pack_frame, variable=self.stringvarTextTranslationOption,values=TextTranslationOption)
		self.target_TextTranslationOption_dropdown.pack(pady=5)
		self.target_TextTranslationOption_dropdown.set(TextTranslationOption[0])
		self.stringvarTextTranslationOption.trace('w', self.Update_Gui) 

		self.label_source_AudioFileLang = customtkinter.CTkLabel(pack_frame, text="For online or Hybrid Translation:\n Select Source Audio file Language:",fg_color="#222121",text_color='#222121', font=("Arial", 12, "bold"))
		self.label_source_AudioFileLang.pack(pady=5)
		
		self.Src_lang = {
			"English": "en",
			"Spanish": "es",
			"French": "fr",
			"German": "de",
			"Japanese": "ja",
			"Korean": "ko",
			"Turkish": "tr",
			"Arabic": "ar",
			"Russian": "ru",
			"Hebrew": "iw",
			"Hindi": "hi",
			"Italian": "it",
			"Portuguese": "pt",
			"Chinese (Mandarin)": "zh",
			"Czech": "cs",
			"Dutch": "nl",
			"Polish": "pl"
			}

		self.stringvarsource_AudioFileLang = customtkinter.StringVar()
		self.source_AudioFileLang_dropdown = customtkinter.CTkOptionMenu(pack_frame, variable=self.stringvarsource_AudioFileLang,fg_color="#222121",text_color='#222121', values=list(self.Src_lang.keys()))
		self.source_AudioFileLang_dropdown.pack(pady=5)
		self.stringvarsource_AudioFileLang.set("Hebrew")
		self.stringvarsource_AudioFileLang.trace('w', self.showErrorIfUserUseLocalTranslationAndSelectSourceLanguage) 

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
		
		self.languages = {
			"English": "en",
			"Spanish": "es",
			"French": "fr",
			"German": "de",
			"Japanese": "ja",
			"Korean": "ko",
			"Turkish": "tr",
			"Arabic": "ar",
			"Russian": "ru",
			"Hebrew": "hu",
			"Hindi": "hi",
			"Italian": "it",
			"Portuguese": "pt",
			"Chinese (Mandarin)": "zh",
			"Czech": "cs",
			"Dutch": "nl",
			"Polish": "pl"
			}

		self.translator_instance = CustomTranslator()

		self.stringvarlanguage = customtkinter.StringVar()
		self.target_language_dropdown = customtkinter.CTkOptionMenu(pack_frame, variable=self.stringvarlanguage, values=list(self.languages.keys()))
		self.target_language_dropdown.pack(pady=5)
		self.stringvarlanguage.set("Arabic")

		self.translate_button = customtkinter.CTkButton(pack_frame, text="Translate", command=self.translate)
		self.translate_button.pack(pady=5)
		
		self.switch_var = customtkinter.StringVar(value="on")
		self.switch_1 = customtkinter.CTkSwitch(master=pack_frame, text="Play translated audio file", command=self.switch_event,variable=self.switch_var, onvalue="on", offvalue="off")
		self.switch_1.pack(padx=20, pady=10)

		self.stop_button = customtkinter.CTkButton(pack_frame, text="Stop Playing Translated File",fg_color="#222121",text_color='#222121',command=self.stop_playing)
		self.stop_button.pack(pady=5)
		
		# Create a frame for widgets using grid
		grid_frame = Frame(master, bg="#222121")
		grid_frame.pack(side="right", padx=20 ,pady=20)
		
		#self.label_translated_text = customtkinter.CTkLabel(grid_frame, text="Translated Text:", font=("Arial", 16, "bold"), text_color="white")
		#self.label_translated_text.grid(row=5, column=0, columnspan=2, pady=10)
		
		self.clear_button = customtkinter.CTkButton(grid_frame, text="Clear",fg_color="#222121",text_color='#222121', command=self.clear_text)
		self.clear_button.grid(row=6, column=0, columnspan=1, pady=10)
		
		self.text_translated = tk.Text(grid_frame, height=20, width=45, wrap = 'word')
		self.text_translated.grid(row=7, column=0, columnspan=1, pady=10)

		self.save_button = customtkinter.CTkButton(grid_frame, text="Save",fg_color="#222121",text_color='#222121', command=self.save_translation)
		self.save_button.grid(row=9, column=0, columnspan=1, pady=10)
		
		#self.progress_bar = customtkinter.CTkProgressBar(grid_frame, variable=DoubleVar(), mode='indeterminate')
		#self.progress_bar.grid(row=11, column=0, columnspan=2, pady=10)

		self.label_status = customtkinter.CTkLabel(grid_frame, text="")
		self.label_status.grid(row=12, column=0, columnspan=2, pady=5)	

	def showErrorIfUserUseLocalTranslationAndSelectSourceLanguage(self,a,b,c):
		if self.target_TextTranslationOption_dropdown.get() == 'Local':
			CTkMessagebox(title="Error", message="With Local Translation: \n You don't need to select audio file Srource Language", icon="cancel")

	def Update_Gui(self,local,online,hybrid):
		if self.target_TextTranslationOption_dropdown.get() != 'Local':
			self.source_AudioFileLang_dropdown.configure(fg_color="#2B7FA3",text_color='white')
			self.label_source_AudioFileLang.configure(fg_color="#222121",text_color='white')
		else:
			self.source_AudioFileLang_dropdown.configure(fg_color="#222121",text_color='#222121')
			self.label_source_AudioFileLang.configure(fg_color="#222121",text_color='#222121')

	def switch_event(self):
		print("switch toggled, current value:", self.switch_var.get())		

	def translate(self):
		if self.audio_path:
			output_path = filedialog.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 Files", "*.mp3")])
			if output_path:
				translation_thread = threading.Thread(target=self.run_translation, args=(output_path,))
				translation_thread.start()
				#self.progress_bar.start()
				self.label_status.configure(text="Translation in progress...",font=("Arial", 16, "bold"),text_color="red")	 
				
	def extract_audio(self):
		input_video = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])

		if input_video != '':
			input_video_file = input_video.split("/")[-1]
			input_video_file = str(input_video_file).replace('.mp4', '')
			output_audio = f"{input_video_file}.mp3"

			command = [
				'ffmpeg',
				'-i', input_video,
				'-vn',
				'-ac', '2',
				'-ar', '44100',
				'-ab', '192k',
				'-f', 'mp3',
				output_audio
			]

			# Run the command in a separate thread
			threading.Thread(target=self.run_ffmpeg_command, args=(command, output_audio)).start()

	def run_ffmpeg_command(self, command, output_audio):
		try:
			# Use subprocess.Popen instead of subprocess.run
			process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			output, error = process.communicate()

			if process.returncode == 0:
				print(f"Conversion successful: {output_audio}")
				messagebox.showinfo("Info", f"Conversion successful: {output_audio}")
			else:
				print(f"Error during conversion: {error.decode('utf-8')}")
				messagebox.showerror("Error", f"Error during conversion: {error.decode('utf-8')}")

		except Exception as e:
			print(f"An error occurred: {str(e)}")
			messagebox.showerror("Error", f"An error occurred: {str(e)}")
	
	def VideoTextAdder(self):	
		def runVideoTextAdder():
			VideoTextAdder_subprocess = subprocess.run(["python", VideoTextAdder_path], check=True)

		VideoTextAdder_path = r'OpenTranslator\VideoTextAdder.py'
		if os.path.exists(VideoTextAdder_path):
			VideoTextAdder_thread = threading.Thread(target=runVideoTextAdder)
			VideoTextAdder_thread.start()

	def AudioReplacerGUI(self):
		def runAudioReplacerGUI():
			AudioReplacerGUI_subprocess = subprocess.run(["python", AudioReplacerGUI_path], check=True)

		AudioReplacerGUI_path = r'OpenTranslator\ReplaceVideoAudio.py'
		if os.path.exists(AudioReplacerGUI_path):
			AudioReplacerGUI_thread = threading.Thread(target=runAudioReplacerGUI)
			AudioReplacerGUI_thread.start()		

	def YouTubeDownloader(self):
		def runYt():
			YouTubeDownloader_subprocess = subprocess.run(["python", YouTubeDownloader_path], check=True)

		YouTubeDownloader_path = r'OpenTranslator\youtube_downloader.py'
		if os.path.exists(YouTubeDownloader_path):
			YouTubeDownloader_thread = threading.Thread(target=runYt)
			YouTubeDownloader_thread.start()
			
		else:
			messagebox.showinfo("YouTubeDownloader_path Not Found")	

	def PyTranscriber(self):
		def RunPyTranscriber():
			pytranscriber_path = r'C:\Program Files (x86)\pyTranscriber\pyTranscriber.exe'
			# Check if pyTranscriber exists
			if os.path.exists(pytranscriber_path):
				subprocess.run([pytranscriber_path])
			else:
				# Show message box to install pyTranscriber
				messagebox.showinfo("PyTranscriber Not Found",
				"Please install pyTranscriber from: https://pytranscriber.github.io/download/")	
				webbrowser.open_new(r'https://pytranscriber.github.io/download/')
		threading.Thread(target=RunPyTranscriber).start()
				
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
				#convert_to_mp3(input_file, output_file)
				threading.Thread(target=convert_to_mp3, args=(input_file, output_file)).start()
			else:
				print("The input file is already an MP3.")
				messagebox.showinfo("Error", "The input file is already an MP3.")
		
		Input_file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.*")])
		
		if Input_file_path != '':
			Start(Input_file_path)
	
	def open_link(self):
		webbrowser.open("https://github.com/overcrash66/OpenTranslator")

	def show_about(self):
		msg = CTkMessagebox(title="About",message = "Open Translator v1.0.0\n\nCreated by Wael Sahli\n\n",option_1="Visite our website",option_2="Close")	
		if msg.get()=='Visite our website':
			self.open_link()
	
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
		self.label_file_title.configure(text=f"")
		self.save_button.configure(fg_color="#222121",text_color='#222121')
		self.clear_button.configure(fg_color="#222121",text_color='#222121')
		self.stop_button.configure(fg_color="#222121",text_color='#222121')
	
	def run_translation(self, output_path):
		input_file = self.audio_path
		self.save_button.configure(fg_color="#222121",text_color='#222121')
		self.clear_button.configure(fg_color="#222121",text_color='#222121')
		self.stop_button.configure(fg_color="#222121",text_color='#222121')
		# Get the duration of the input audio file
		ffprobe_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input_file}"'
		input_duration = float(subprocess.check_output(ffprobe_cmd, shell=True))
		
		if input_duration > 30 :
			# Set the maximum duration for each chunk (30 seconds in this case)
			max_chunk_duration = 30

			# Calculate the number of chunks required
			num_chunks = int(input_duration / max_chunk_duration)
			#if num_chunks > 0:

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
				if current_progress >= 2:
					current_progress = current_progress - 1
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
																 self.languages[self.stringvarlanguage.get()],self.Src_lang[self.stringvarsource_AudioFileLang.get()],
																 chunk_idx, output_path,self.target_TextTranslationOption_dropdown.get())											 
				except Exception as e:
					print(f"{e}")
					#self.progress_bar.stop()
					self.label_status.configure(text="An Error occurred!",font=("Arial", 16, "bold"),text_color="red")
				
				chunk_files.append(chunk_output_path)
				
				self.text_translated.configure(state='normal')
				self.text_translated.insert('end', f"{translation_result}\n\n")
				self.text_translated.configure(state='disabled')

				Translation_chunk_output_path = f"{output_path}_Translation_chunk{chunk_idx + 1}.wav"
				Translation_chunk_files.append(Translation_chunk_output_path)
			

			# Merge individual chunk files into the final output file
			final_output_path = f"{output_path}-temp.wav"
			if self.target_TextTranslationOption_dropdown.get() == 'Local' or self.target_TextTranslationOption_dropdown.get() == 'Hybrid':
				self.merge_audio_files(Translation_chunk_files, final_output_path)
			if self.target_TextTranslationOption_dropdown.get() == 'Online':
				self.merge_online_audio_files(Translation_chunk_files, final_output_path)

			subprocess.run(['ffmpeg', '-i', final_output_path, '-codec:a', 'libmp3lame', output_path], check=True)
			os.remove(final_output_path)

			if self.switch_var.get() == 'on':
				# Play the final merged audio file
				self.translator_instance.play_audio(output_path)
				self.stop_button.configure(fg_color="#2B7FA3",text_color='white')

			# Cleanup: Delete individual chunk files
			self.delete_chunk_files(chunk_files)
			self.delete_chunk_files(Translation_chunk_files)
			chunk_files = []  # List to store individual chunk files
			Translation_chunk_files = []
			#self.progress_bar.stop()

			self.label_status.configure(text="Translation complete!",font=("Arial", 16, "bold"),text_color="green")
			
			self.save_button.configure(fg_color="#2B7FA3",text_color='white')
			self.clear_button.configure(fg_color="#2B7FA3",text_color='white')	

		if input_duration <= 30 and self.target_TextTranslationOption_dropdown.get() == 'Local':
			print("Audio File less or equal 30 sec !")	
			self.save_button.configure(fg_color="#222121",text_color='#222121')
			self.clear_button.configure(fg_color="#222121",text_color='#222121')
			self.stop_button.configure(fg_color="#222121",text_color='#222121')
			# Update label text
			self.label_status.configure(
				text=f"Translation in progress...",
				font=("Arial", 16, "bold"), text_color="red"
			)

			# Use a consistent naming pattern for chunk files
			chunk_output_path = f"{input_file}"
			chunk_idx = 0
			try:
				translation_result = self.translator_instance.process_audio_chunk(chunk_output_path,
															 self.languages[self.stringvarlanguage.get()],self.Src_lang[self.stringvarsource_AudioFileLang.get()],
															 chunk_idx, output_path,self.target_TextTranslationOption_dropdown.get())											 
			except Exception as e:
				print(f"{e}")
				#self.progress_bar.stop()
				self.label_status.configure(text="An Error occurred!",font=("Arial", 16, "bold"),text_color="red")
			
			chunk_files = []  # List to store individual chunk files
			Translation_chunk_files = []		
			chunk_files.append(chunk_output_path)
			
			self.text_translated.configure(state='normal')
			self.text_translated.insert('end', f"{translation_result}\n\n")
			self.text_translated.configure(state='disabled')

			Translation_chunk_output_path = f"{output_path}_Translation_chunk1.wav"
			Translation_chunk_files.append(Translation_chunk_output_path)
			
			subprocess.run(['ffmpeg', '-i', Translation_chunk_output_path, '-codec:a', 'libmp3lame', output_path], check=True)
			os.remove(Translation_chunk_output_path)

			if self.switch_var.get() == 'on':
				# Play the final merged audio file
				self.translator_instance.play_audio(output_path)
				self.stop_button.configure(fg_color="#2B7FA3",text_color='white')

			#self.progress_bar.stop()

			self.label_status.configure(text="Translation complete!",font=("Arial", 16, "bold"),text_color="green")	

			self.save_button.configure(fg_color="#2B7FA3",text_color='white')
			self.clear_button.configure(fg_color="#2B7FA3",text_color='white')		
		
		if input_duration <= 30 and self.target_TextTranslationOption_dropdown.get() == 'Online' or input_duration <= 30 and self.target_TextTranslationOption_dropdown.get() == 'Hybrid':
			#self.progress_bar.stop()
			print('For online translation: you need to use an audio file longer then 30 sec !')	
			messagebox.showerror("Error", f"For online translation: you need to use an audio file longer then 30 sec !")
			self.label_status.configure(text="",font=("Arial", 16, "bold"),text_color="black")

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

	def merge_online_audio_files(self, input_files, output_file):
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
		self.stop_button.configure(fg_color="#222121",text_color='#222121')
	
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