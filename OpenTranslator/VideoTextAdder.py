import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import threading
import customtkinter
import ctypes
import sys
if sys.platform == "win32":
    ctypes.windll.user32.SetProcessDPIAware()
customtkinter.set_appearance_mode("System")    # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class VideoTextAdder:
    def load_default_config(self):
        try:
            with open('config.json', 'r') as config_file:
                config = json.load(config_file)
            return config
        except FileNotFoundError:
            return {
                'text_color': 'red',
                'font_size': 24,
                'font_style': 'bold',
                'default_text': 'Voice translated and generated using AI'
            }

    def apply_text_to_video(self):
        input_video_path = self.entry_video_path.get()
        output_video_path = self.entry_output_path.get()
        text_to_add = self.entry_text.get()
        text_color = self.entry_text_color.get()
        font_size = int(self.entry_font_size.get())
        font_style = self.entry_font_style.get()

        # Run add_text_to_video in a separate thread to avoid freezing the GUI
        threading.Thread(target=self.add_text_to_video, args=(input_video_path, output_video_path, text_to_add, text_color, font_size, font_style)).start()

    def save_config(self, config):
        with open('config.json', 'w') as config_file:
            json.dump(config, config_file)

    def browse_video(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        self.entry_video_path.delete(0, tk.END)
        self.entry_video_path.insert(0, self.file_path)

    def browse_output(self):
        self.file_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("Video files", "*.mp4")])
        self.entry_output_path.delete(0, tk.END)
        self.entry_output_path.insert(0, self.file_path)

    def add_text_to_video(self, input_video, output_video, text, text_color='white', font_size=20, font_style='regular'):
        try:
            command = [
                'ffmpeg',
                '-i', input_video,
                '-vf', f"drawtext=text='{text}':x=(w-tw-10):y=10:fontsize={font_size}:fontcolor={text_color}",
                '-c:a', 'copy',
                output_video
            ]
            
            subprocess.run(command)
            
            print(f"Video with text added successfully: Output path: {output_video}")
            messagebox.showinfo("Video Text Added", f"Video with text added successfully!\nOutput path: {output_video}")

        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def __init__(self):
        window = customtkinter.CTk()
        window.title("Video Text Adder")
        window.geometry("600x280")
        window.resizable(False, False)
        window.attributes("-topmost", True)
        default_config = self.load_default_config()

        customtkinter.CTkLabel(window, text="Input Video:").grid(row=0, column=0, padx=10, pady=5)
        self.entry_video_path = customtkinter.CTkEntry(window, width=300)
        self.entry_video_path.grid(row=0, column=1, padx=10, pady=5)
        customtkinter.CTkButton(window, text="Browse", command=self.browse_video).grid(row=0, column=2, pady=5)

        customtkinter.CTkLabel(window, text="Output Video:").grid(row=1, column=0, padx=10, pady=5)
        self.entry_output_path = customtkinter.CTkEntry(window, width=300)
        self.entry_output_path.grid(row=1, column=1, padx=10, pady=5)
        customtkinter.CTkButton(window, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=5)

        customtkinter.CTkLabel(window, text="Text to Add:").grid(row=2, column=0, padx=10, pady=5)
        self.entry_text = customtkinter.CTkEntry(window, width=300)
        self.entry_text.grid(row=2, column=1, padx=10, pady=5)
        self.entry_text.insert(0, default_config['default_text'])

        customtkinter.CTkLabel(window, text="Text Color:").grid(row=3, column=0, padx=10, pady=5)
        self.entry_text_color = customtkinter.CTkEntry(window, width=50)
        self.entry_text_color.grid(row=3, column=1, padx=10, pady=5)
        self.entry_text_color.insert(0, default_config['text_color'])

        customtkinter.CTkLabel(window, text="Font Size:").grid(row=4, column=0, padx=10, pady=5)
        self.entry_font_size = customtkinter.CTkEntry(window, width=50)
        self.entry_font_size.grid(row=4, column=1, padx=10, pady=5)
        self.entry_font_size.insert(0, str(default_config['font_size']))

        customtkinter.CTkLabel(window, text="Font Style:").grid(row=5, column=0, padx=10, pady=5)
        self.entry_font_style = tk.Entry(window, width=10)
        self.entry_font_style.grid(row=5, column=1, padx=10, pady=5)
        self.entry_font_style.insert(0, default_config['font_style'])

        customtkinter.CTkButton(window, text="Apply Text to Video", command=self.apply_text_to_video).grid(row=6, column=1, pady=10)

        self.entry_video_path.insert(0, "")
        self.entry_output_path.insert(0, "")

        window.mainloop()

if __name__ == "__main__":
    gui = VideoTextAdder()
