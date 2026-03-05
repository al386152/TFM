import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
from pathlib import Path

import torch.nn as nn

DEFAULT_BASE_NAME = "Dectector de RetinopatÍA - CNN" #"RetCNN"
DEFAULT_IMAGE_TEXT = "RETINOPATHY DEGREE:"
DEFAULT_IMAGE_STATE = f"{DEFAULT_IMAGE_TEXT} Retinography not processed."
DEFAULT_IMAGE_TEXT = "Processing: "
IMAGE_SIZE = (400, 400)
WINDOW_SIZE = (IMAGE_SIZE[0] + 100, IMAGE_SIZE[1] + 200)


# TODO: Hacer un botón para procesar una imagen [hacer que se bloquee hasta que esté completado]
# TODO: Hacer que se pare todo de alguna forma (?)
# TODO: Añadir texto para cada imágen indicando el estado
# TODO: Hacer un botón para procesar todas las imágenes
# TODO: Hacer un log.

class Tab(ttk.Frame):
    def __init__(self, master, the_gui: "TheGUI", file: str) -> None:
        super().__init__(master)
        self.main_tab:ttk.Notebook = master
        self.the_gui = the_gui
        self.img: ImageTk.PhotoImage
        self.img_label: tk.Label
        self.text_state: tk.Label
        self.remove_button: tk.Button
        self.process_button: tk.Button

        img = Image.open(file)
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        self.img = ImageTk.PhotoImage(img)

        self.img_label = tk.Label(self, image = self.img)
        self.img_label.pack(side="top")

        self.text_state = tk.Label(self, text = DEFAULT_IMAGE_STATE)
        self.text_state.pack(side="bottom")
        
        def remove_file(_file = file):
            self.the_gui.close_file(_file)
        self.remove_button = tk.Button(self, text ='Remove retinography', command = remove_file)
        self.remove_button.pack(side="bottom")

        def _processing_images(_files = file, tab = self):
            files = [(_files, tab)]
            processing_images(list_images=files)

        self.process_button = tk.Button(self, text ='Process retinography', command = _processing_images)
        self.process_button.pack(side="bottom")

        self.main_tab.add(self, text=Path(file).stem)
        self.the_gui.files[file] = self
# ----

class TheGUI(tk.Frame):

    def __init__(self, model: nn.Module, master=None):
        super().__init__(master)
        self.pack()
        self.cnn_model = model
        self.files: dict[str, Tab] = dict()

        self.main_tab = ttk.Notebook(self)

        def _processing_images(files = self.files):
            processing_images(files)

        self.button_remove_all = tk.Button(self, text ='Process all retinographies', command = _processing_images)
        self.button_remove_all.pack(side="bottom", expand=False)

        self.button_remove_all = tk.Button(self, text ='Remove all retinographies', command = self.close_files, state="disabled")
        self.button_remove_all.pack(side="bottom", expand=False)

        button_open = tk.Button(self, text ='Add retinographies', command = self.open_images)
        button_open.pack(side="bottom", expand=False)
    # ----


    def enable_remove_all_button(self):
        if len(self.files.keys()) == 0:
            self.button_remove_all["state"] = "disabled"
        else:
            self.button_remove_all["state"] = "active"

    def open_images(self):
        files = filedialog.askopenfilenames()
        for file in files:
            if file not in self.files:
                Tab(self.main_tab, self, file)

        self.enable_remove_all_button()
        self.main_tab.pack(side="top")
    # ----

    def close_files(self):
        print(f"==> close_files || files to close: {self.files.keys()=}")
        files_to_remove = list(self.files.keys())
        for file in files_to_remove:
            print(f"{file=}")
            self.close_file(file)
    # ----

    def close_file(self, file: str):
        print(f"==> close_file: {file}")
        tab = self.files[file]
        del tab.img
        del self.files[file]
        tab.forget()
        tab.destroy()
        self.main_tab.pack(side="top")
        self.enable_remove_all_button()


########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################

# TODO: REMOVE
import time

def doing_the_inference():
    # NOTE: Parece que tengo que lanzar esto en un hilo a parte
    time.sleep(2)
    print(f"doing_the_inference")

def intializing_dataset():
    print(f"intializing_dataset")

def processing_images(list_images: list[tuple[str, Tab]]):
    print(f"processing_images -> {str([img[0] for img in list_images])}")
    
    # TODO: [mirar de cómo hacer esto]

    # Blocking the "process" and "delete" button.
    for _, tab in list_images:
        tab.remove_button["state"] = "disabled"
        tab.process_button["state"] = "disabled"
    
    intializing_dataset()
    doing_the_inference()
    
    # Unlocking the "process" and "delete" button.
    for _, tab in list_images:
        tab.remove_button["state"] = "active"
        tab.process_button["state"] = "active"


def initialize_model() -> nn.Module:
    print("Initializing the model")
    # TODO Hacer que se cargue y se inicialice el modelo.
    return nn.Module()

def startGui():
    model = initialize_model()
    root = TheGUI(model)
    root.master.title(DEFAULT_BASE_NAME)
    root.master.minsize(*WINDOW_SIZE)
    #root.master.maxsize(600, 700)
    root.mainloop()
# ----


if __name__ == "__main__":
    startGui()