from tkinter import filedialog, messagebox
import threading
import subprocess
import customtkinter
import os

class AudioReplacerGUI:
    def __init__(self):
        new_window = customtkinter.CTk()
        new_window.title("Audio Replacer")
        new_window.geometry("300x220")
        new_window.minsize(300, 220)
        new_window.maxsize(300, 240)
        new_window.attributes('-fullscreen', False)
        new_window.attributes("-topmost", True)
        self.video_path = None
        self.audio_path = None
        self.output_path = None
        self.thread = None  # Thread variable to keep track of the background task

        self.label = customtkinter.CTkLabel(new_window, text="Select Video, Audio, and Output Paths:",
                                            font=("Arial", 12, "bold"), text_color="green")
        self.label.pack(pady=5)

        self.video_button = customtkinter.CTkButton(new_window, text="Select Video", command=self.select_video)
        self.video_button.pack(pady=5)

        self.audio_button = customtkinter.CTkButton(new_window, text="Select Audio", command=self.select_audio)
        self.audio_button.pack(pady=5)

        self.output_button = customtkinter.CTkButton(new_window, text="Select Output", command=self.select_output)
        self.output_button.pack(pady=5)

        # Replace Audio
        self.replace_button = customtkinter.CTkButton(new_window, text="Run", command=self.replace_audio)
        self.replace_button.pack(pady=5)
        new_window.mainloop()

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
        def get_duration(file_path):
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
        video_duration = get_duration(video_path)
        audio_duration = get_duration(audio_path)

        if video_duration is None or audio_duration is None:
            print("Error getting duration information.")
            return None

        speed_factor = audio_duration / video_duration

        if 0.5 <= speed_factor <= 2:
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

    def replace_audio_async(self):
        # This function will be called in a separate thread
        self.change_text_color(self.replace_button)

        if self.video_path and self.audio_path and self.output_path:
            # Check if files exist
            if not all(map(os.path.exists, [self.video_path, self.audio_path])):
                messagebox.showerror("Error", "Video or audio file not found.")
                return

            adjusted_audio = self.change_speed(self.audio_path, self.video_path)

            if adjusted_audio:
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
                    messagebox.showerror("Error", "Error during video conversion.")

                os.remove("output_speeded_up.mp3")
                messagebox.showinfo("Info", f"Conversion successful !")

    def replace_audio(self):
        # Start a new thread for the background task
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("Info", "Conversion is already in progress.")
            return

        self.thread = threading.Thread(target=self.replace_audio_async)
        self.thread.start()


if __name__ == "__main__":
    gui = AudioReplacerGUI()
