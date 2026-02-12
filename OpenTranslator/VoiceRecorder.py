import os, sys
import tkinter as tk
from tkinter import messagebox, simpledialog
from pydub import AudioSegment
from scipy.io.wavfile import write
import customtkinter
import sounddevice as spd

fs = 44100  # Sample rate
seconds = 5  # Duration of recording

class AudioRecorderGUI:
    def __init__(self):
        self.setup_gui()

    def setup_gui(self):
        self.Recorder_Gui = customtkinter.CTk()
        self.Recorder_Gui.title("Mic Recorder")
        self.Recorder_Gui.geometry("300x150")
        self.Recorder_Gui.resizable(False, False)
        self.Recorder_Gui.attributes("-topmost", True)

        self.record_button = customtkinter.CTkButton(self.Recorder_Gui, text="Record", command=self.record)
        self.record_button.pack(pady=5)

        self.stop_button = customtkinter.CTkButton(self.Recorder_Gui, text="Stop Recording", command=self.stop)
        self.stop_button.pack(pady=20)
        self.stop_button.configure(state="disabled")

        self.label_input = customtkinter.CTkLabel(self.Recorder_Gui, text="", font=("Arial", 12, "bold"),text_color="white")
        self.label_input.pack(pady=5)

        self.Recorder_Gui.mainloop()
    def record(self):
        self.stop_button.configure(state="normal")
        self.record_button.configure(state="disabled")
        self.output_file = "output.wav"

        print("Recording...")
        self.label_input.configure(self.Recorder_Gui, text=f"Recording...",font=("Arial", 16, "bold"), text_color="red")

        self.myrecording = spd.rec(int(seconds * fs), samplerate=fs, channels=2)

    def stop(self):
        self.stop_button.configure(state="disabled")
        self.record_button.configure(state="normal")
        spd.stop()
        write('output.wav', fs, self.myrecording)  # Save as WAV file

        mp3_output_file = simpledialog.askstring("Output File Name", "Enter the name of the output MP3 file:")
        mp3_output_file = str(mp3_output_file)+'.mp3'
        if mp3_output_file:
            convert_to_mp3(self.output_file, mp3_output_file)

            messagebox.showinfo("Success", f"Audio saved as {mp3_output_file}")
            self.label_input.configure(self.Recorder_Gui, text=f"",font=("Arial", 16, "bold"), text_color="red")

def convert_to_mp3(input_file, output_file):
    sound = AudioSegment.from_wav(input_file)
    sound.export(output_file, format="mp3")
    os.remove(input_file)

if __name__ == "__main__":
    gui = AudioRecorderGUI()
