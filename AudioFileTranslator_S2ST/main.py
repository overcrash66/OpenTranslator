# AudioFileTranslator_S2ST/main.py

from tkinter import Tk
from AudioFileTranslator_S2ST.translator_gui import TranslatorGUI
import customtkinter

def run_gui():
    root = customtkinter.CTk()
    app = TranslatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
