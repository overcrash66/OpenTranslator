from tkinter import Tk, Label, Button, filedialog, messagebox
import os, subprocess
from CTkMenuBar import *
import customtkinter

class AudioReplacerGUI:
	def change_text_color(self, btn):
		new_color = 'green'
		btn.configure(fg_color=new_color)

	def select_video(self):
		self.change_text_color(self.video_button)
		self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])

	def select_audio(self):
		self.change_text_color(self.audio_button)
		self.audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])

	def select_output(self):
		self.change_text_color(self.output_button)
		self.output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("Video Files", "*.mp4")])
	
	def change_speed(self, audio_path, video_path):
		def get_audio_duration(file_path):
			try:
				# Run ffprobe command to get audio information
				command = [
					'ffprobe',
					'-v', 'error',
					'-show_entries', 'format=duration',
					'-of', 'default=noprint_wrappers=1:nokey=1',
					file_path
				]
				result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
				
				# Parse duration from the result
				duration = float(result.strip())
				
				return duration
			except subprocess.CalledProcessError as e:
				print(f"Error while running ffprobe: {e.output}")
				return None
			
		print("Starting change_speed")
		ffprobe_command = [
		'ffprobe',
		'-v', 'error',
		'-show_entries', 'format=duration',
		'-of', 'default=noprint_wrappers=1:nokey=1',
		video_path
		]
		
		video_duration = float(subprocess.check_output(ffprobe_command))
		audio_clip = get_audio_duration(audio_path)
		
		# Calculate the speed factor to match the video duration
		speed_factor = audio_clip / video_duration
		
		if speed_factor >= 0.5 and speed_factor <=2:
			speed_factor = speed_factor
		else:
			speed_factor = 1
			
		New_Audio_output_path = 'output_speeded_up.mp3'
		
		# Use FFmpeg to speed up or slow down the audio
		ffmpeg_command = [
			'ffmpeg',
			'-i', audio_path,
			'-filter:a', f'atempo={speed_factor}',
			New_Audio_output_path
		]
		subprocess.run(ffmpeg_command, check=True)
		
		changed_audio = New_Audio_output_path
		return changed_audio

	def replace_audio(self):
		self.change_text_color(self.replace_button)
		if self.video_path and self.audio_path and self.output_path:

			adjusted_audio = self.change_speed(self.audio_path, self.video_path)
			input_video = self.video_path
			input_audio = adjusted_audio
			output_video = self.output_path
			try:
				# FFmpeg command to replace audio in a video
				command = [
					'ffmpeg',
					'-i', input_video,
					'-i', input_audio,
					'-c:v', 'copy',
					'-c:a', 'aac',
					'-strict', 'experimental',
					'-map', '0:v:0',
					'-map', '1:a:0',
					output_video
				]

				# Run the FFmpeg command
				subprocess.run(command, check=True)

				print(f"Audio replaced successfully. Output video saved as {output_video}")
			except subprocess.CalledProcessError as e:
				print(f"Error while running FFmpeg: {e.stderr}")
			
			os.remove("output_speeded_up.mp3")
			messagebox.showinfo("Info", f"Conversion successful !")

	def __init__(self):
		new_window = customtkinter.CTk()
		new_window.title("Audio Replacer")
		new_window.geometry("300x220")
		new_window.minsize(300,220)
		new_window.maxsize(300,240)
		new_window.attributes('-fullscreen', False)
		new_window.attributes("-topmost", True)
		self.video_path = None
		self.audio_path = None
		self.output_path = None

		self.label = customtkinter.CTkLabel(new_window, text="Select Video, Audio, and Output Paths:", font=("Arial", 12, "bold"),text_color="green")
		self.label.pack(pady=5)

		self.video_button = customtkinter.CTkButton(new_window, text="Select Video", command=self.select_video)
		self.video_button.pack(pady=5)

		self.audio_button = customtkinter.CTkButton(new_window, text="Select Audio", command= self.select_audio)
		self.audio_button.pack(pady=5)

		self.output_button = customtkinter.CTkButton(new_window, text="Select Output", command=self.select_output)
		self.output_button.pack(pady=5)
		
		#Replace Audio
		self.replace_button = customtkinter.CTkButton(new_window, text="Run", command=self.replace_audio)
		self.replace_button.pack(pady=5)
		new_window.mainloop()	

if __name__ == "__main__":
	gui = self.AudioReplacerGUI()
	
