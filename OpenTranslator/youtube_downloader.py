from tkinter import filedialog, messagebox
from pytube import YouTube
from PIL import Image, ImageTk
import customtkinter
import subprocess
import re
import os
import threading
import unicodedata

class YouTubeDownloader:
    def __init__(self):
        self.setup_gui()

    def setup_gui(self):
        new_window = customtkinter.CTk()
        new_window.title("YouTube Downloader")
        new_window.geometry("500x150")
        new_window.resizable(False, False)
        new_window.attributes("-topmost", True)

        url_label = customtkinter.CTkLabel(new_window, text="YouTube URL:")
        url_label.grid(row=0, column=0, padx=10, pady=10)

        self.url_entry = customtkinter.CTkEntry(new_window, width=300)  # Set width to 300
        self.url_entry.grid(row=0, column=1, padx=10, pady=10)

        self.download_button = customtkinter.CTkButton(new_window, text="Download", command=self.start_download)
        self.download_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.status_label = customtkinter.CTkLabel(new_window, text="", wraplength=400, justify="left")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=10)

        new_window.mainloop()

    def sanitize_filename(self, title):
        title = re.sub(r'\W+', '_', title)
        title = unicodedata.normalize('NFKD', title).encode('ASCII', 'ignore').decode('utf-8')
        return title


    def start_download(self):
        url = self.url_entry.get()

        if not url:
            messagebox.showerror("Error", "Please enter a valid YouTube URL.")
            return

        download_thread = threading.Thread(target=self.download, args=(url,))
        download_thread.start()

    def download(self, url):
        try:
            yt = YouTube(url)
            video_title = self.sanitize_filename(yt.title)
            output_path = os.path.join(os.getcwd(), f"{video_title}.mp4")

            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"Deleted existing file: {output_path}")

            self.status_label.configure(text="Downloading...")

            video_stream = yt.streams.get_highest_resolution()
            video_url = video_stream.url

            subprocess.run(['ffmpeg', '-i', video_url, '-c', 'copy', output_path])

            message = f"Download complete!\nFile saved as: {output_path}"
            self.status_label.configure(text=message)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            self.status_label.configure(text=error_message)

if __name__ == "__main__":
    youtube_downloader_gui = YouTubeDownloader()
