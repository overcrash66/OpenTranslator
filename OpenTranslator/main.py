from tkinter import Tk
from OpenTranslator.translator_gui import TranslatorGUI
import customtkinter

def run_gui():
    root = customtkinter.CTk()
    app = TranslatorGUI(root)
    root.mainloop()

