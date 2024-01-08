from tkinter import filedialog, messagebox
from pytube import YouTube
from PIL import Image, ImageTk
import customtkinter
import subprocess
import re, os

class YouTubeDownloader:
    def sanitize_filename(self, title):
        # Replace non-alphanumeric characters with underscores
        return re.sub(r'\W+', '_', title)

    def download(self):
        url = self.url_entry.get()

        if not url:
            messagebox.showerror("Error", "Please enter a valid YouTube URL.")
            return

        try:
            yt = YouTube(url)
            video_title = self.sanitize_filename(yt.title)
            output_path = f"{video_title}.mp4"
            current_path = os.getcwd()
            
            t = current_path + '\\'+ output_path
            if os.path.exists(t):
                t = output_path
                os.remove(t)
                print(f"deleted file {t}")

            self.status_label.configure(text="Downloading...")
            
            # Get the highest resolution stream
            video_stream = yt.streams.get_highest_resolution()
            # Get the video stream URL
            video_url = video_stream.url
            # Use subprocess to call ffmpeg for downloading
            subprocess.run(['ffmpeg', '-i', video_url, '-c', 'copy', f'{output_path}'])

            self.status_label.configure(text=f"Download complete!\nFile saved as: {output_path}")

        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}")
    def __init__(self):
        new_window = customtkinter.CTk()
        new_window.title("YouTube Downloader")
        new_window.attributes('-fullscreen', False)
        new_window.resizable(False, False)
        new_window.attributes("-topmost", True)
        url_label = customtkinter.CTkLabel(new_window, text="YouTube URL:")
        url_label.grid(row=0, column=0, padx=10, pady=10)

        self.url_entry = customtkinter.CTkEntry(new_window, width=500)
        self.url_entry.grid(row=0, column=1, padx=10, pady=10)

        self.download_button = customtkinter.CTkButton(new_window, text="Download", command=self.download)
        self.download_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.status_label = customtkinter.CTkLabel(new_window, text="")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=10)
        new_window.mainloop() 

if __name__ == "__main__":
    youtube_downloader_gui = YouTubeDownloader()
