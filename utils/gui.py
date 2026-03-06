import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

import torch
import torch.nn as nn

import utils.model_related as mr
import utils.constants as cons
from torchvision.transforms.functional import pil_to_tensor
from collections import OrderedDict

DEFAULT_BASE_NAME = "Dectector de RetinopatÍA - CNN" #"RetCNN"
DEFAULT_IMAGE_TEXT = "RETINOPATHY DEGREE:"
DEFAULT_IMAGE_STATE = f"{DEFAULT_IMAGE_TEXT} Retinography not processed."
DETECTION_IN_PROGRESS_TEXT = "Detection in progress"
IMAGE_SIZE = (524, 524)
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

        self.pil_img = Image.open(file).convert("RGB")
        self.pil_img = self.pil_img.resize(IMAGE_SIZE, Image.LANCZOS)
        self.img = ImageTk.PhotoImage(self.pil_img)

        self.img_label = tk.Label(self, image = self.img)
        self.img_label.pack(side="top", expand=True)

        self.text_state = tk.Label(self, text = DEFAULT_IMAGE_STATE)
        self.text_state.pack(side="bottom")
        
        def remove_file(_file = file):
            self.the_gui.close_file(_file)
        self.remove_button = tk.Button(self, text ='Remove retinography', command = remove_file)
        self.remove_button.pack(side="bottom")

        def _processing_images(_files = file, tab = self):
            files = [(_files, tab)]
            processing_images(list_images=files, the_gui=the_gui,
                              thread_pool=self.the_gui.thread_pool,
                              args=self.the_gui.args)

        self.process_button = tk.Button(self, text ='Process retinography', command = _processing_images)
        self.process_button.pack(side="bottom")

        self.main_tab.add(self, text=Path(file).stem)
        self.the_gui.files[file] = self
# ----

class TheGUI(tk.Frame):

    def __init__(self, model: nn.Module, thread_pool: ThreadPoolExecutor, args: dict, master=None):
        super().__init__(master)
        self.pack()
        self.args = args
        self.model = model
        self.button_remove_all: tk.Button
        self.button_process_all: tk.Button
        self.button_open: tk.Button

        self.files: dict[str, Tab] = dict()
        self.thread_pool: ThreadPoolExecutor = thread_pool
        self.main_tab = ttk.Notebook(self)

        def _processing_images(files = self.files, the_gui=self, _thread_pool=self.thread_pool, args = self.args):
            _files = list(files.items())
            processing_images(_files, the_gui=the_gui, thread_pool=_thread_pool, args=args)

        self.button_process_all = tk.Button(self, text ='Process all retinographies', command = _processing_images, state="disabled")
        self.button_process_all.pack(side="bottom", expand=False)

        self.button_remove_all = tk.Button(self, text ='Remove all retinographies', command = self.close_files, state="disabled")
        self.button_remove_all.pack(side="bottom", expand=False)

        self.button_open = tk.Button(self, text ='Add retinographies', command = self.open_images)
        self.button_open.pack(side="bottom", expand=False)
    # ----


    def enable_remove_process_all_buttons(self, force_disable=False, force_enable=False):
        if force_disable:
            self.button_remove_all.config(state="disabled")
            self.button_process_all.config(state="disabled")
        elif force_enable:
            self.button_remove_all.config(state="active")
            self.button_process_all.config(state="active")

        elif len(self.files.keys()) == 0:
            self.button_remove_all.config(state="disabled")
            self.button_process_all.config(state="disabled")
        else:
            self.button_remove_all.config(state="active")
            self.button_process_all.config(state="active")

    def open_images(self):
        files = filedialog.askopenfilenames()
        for file in files:
            if file not in self.files:
                Tab(self.main_tab, self, file)

        self.enable_remove_process_all_buttons()
        self.main_tab.pack(side="top", expand=True)
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
        self.enable_remove_process_all_buttons()


########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################

def inference(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    print(f"doing_the_inference")
    num_elementos = len(dataloader)

    print(f"{type(model)=}")
    #print(f"{model=}")

    model.eval()

    with torch.no_grad():
        for i, (inputs, ) in enumerate(dataloader):
            print(f"Processing element: [{i+1}/{num_elementos}]")
            print(f"{inputs.shape=}")
            inputs: torch.Tensor
            inputs = inputs.to(device)
            outputs = model(inputs)
            print(f"Element {i}: {outputs.shape=}")
    return outputs
# ------


def intializing_dataset(data: list[tuple[str, Tab]]) -> torch.utils.data.DataLoader:
    print(f"intializing_dataset")

    _data = [((pil_to_tensor(tab.pil_img)).float() / 255.0) for _, tab in data]
    print(f"{_data=}")
    tensor_data: torch.Tensor = torch.stack(_data)
    print(f"{tensor_data.shape=}")
    dataset = torch.utils.data.TensorDataset(tensor_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(data))

    return dataloader
    # --


def do_the_inference(model: nn.Module, data: list):
    print(f"{model=}\ndata:{data=}")
    try:
        dataloader = intializing_dataset(data)
        print(f"dataset initialized")
        outputs = inference(model, dataloader)
    except Exception as e:
        from traceback import print_exception
        print_exception(e)
    return outputs

def processing_images(list_images: list[tuple[str, Tab]], the_gui: TheGUI, thread_pool: ThreadPoolExecutor, args: dict):
    print(f"processing_images -> {str([img[0] for img in list_images])}")

    print(f"list_images:\n{list_images}")

    def activate_buttons(resultado_procesamiento):
        print(f"{len(resultado_procesamiento)=} || {len(list_images)=}")
        print("activate_buttons started")
        grado_rd = torch.argmax(resultado_procesamiento, dim=1)
        print(f"{len(grado_rd)=} || {grado_rd.shape=} || {len(list_images)=} || {resultado_procesamiento=}")

        for i, (_, tab) in enumerate(list_images):
            tab.remove_button.config(state = "active")
            tab.process_button.config(state = "active")
            print(f"{i} || {int(grado_rd[i])=} || {cons.RETINOPATHY_NUMBER_TO_GRADE[int(grado_rd[i])]=}")
            tab.text_state.config(text= f"{DEFAULT_IMAGE_TEXT} {cons.RETINOPATHY_NUMBER_TO_GRADE[int(grado_rd[i])]} detected. {100*resultado_procesamiento[i][grado_rd[i]]:.2f} %")
        
        
        the_gui.enable_remove_process_all_buttons()
        the_gui.button_open.config(state="active")
        print("activate_buttons finished")
    # ----

    def _do_inference(model = the_gui.model, _list_images = list_images):
        return do_the_inference(model, _list_images)

    def at_end_processing(promesa: Future) -> None:
        print("at_end_processing started")
        resultado_procesamiento = promesa.result()
        # Unlocking the "process" and "delete" button.
        the_gui.after(0, activate_buttons, resultado_procesamiento)
    # ----

    the_gui.enable_remove_process_all_buttons(force_disable=True)
    the_gui.button_open.config(state="disabled")
    
    # Blocking the "process" and "delete" button.
    print(f"{type(list_images)=} || {len(list_images)} || {list_images=}")
    for _, tab in list_images:
        print(f"{tab=} || {tab.remove_button=} || {tab.process_button=}")
        tab.remove_button.config(state = "disabled")
        tab.process_button.config(state = "disabled")
        tab.text_state.config(text=f"{DETECTION_IN_PROGRESS_TEXT}")

    future_promesa = thread_pool.submit(_do_inference)
    
    future_promesa.add_done_callback(at_end_processing)


def initialize_model(args: dict) -> nn.Module:

    class Model (nn.Module):
        def __init__(self, base_model, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.base_model = base_model
            self.softmax = torch.nn.Softmax()
        
        def forward(self, x) -> torch.Tensor:
            x = self.base_model(x)
            x = self.softmax(x)
            return x

    print("Initializing the model")
    model = mr.load_model(args, torch.device("cpu"), True, False)
    if isinstance(model, OrderedDict):
        weights = OrderedDict({key[len("module."):]: value for key, value in model.items()})
        model: torch.nn.Module = cons.SWITCH_MODELOS[args[cons.MODEL]](**{"num_classes": 5})
        print(f"{weights.keys()=}")
        model.load_state_dict(weights)

    print(f"Model initalized:\n{model}\n==========")
    return Model(model)

def startGui(args):

    args = test_args = {
        cons.MODEL: "densenet169",
        cons.BATCH_SIZE: 1,
        cons.MODEL_WEIGHTS_PATH: "/home/usuario/Documentos/Resultados/pesos/clasificacion/model_densenet169.pth",
        cons.IS_DISTRIBUTED: False
    }

    model = initialize_model(args)
    thread_pool = ThreadPoolExecutor(max_workers = 1)
    root = TheGUI(model, thread_pool, args)
    root.master.title(DEFAULT_BASE_NAME)
    root.master.minsize(*WINDOW_SIZE)
    #root.master.maxsize(600, 700)
    root.mainloop()
    thread_pool.shutdown()
# ----


def main():
    test_args = {
        cons.MODEL: "densenet169",
        cons.BATCH_SIZE: 1,
        cons.MODEL_WEIGHTS_PATH: "/home/usuario/Documentos/Resultados/pesos/clasificacion/model_densenet169.pth"
    }
    startGui(test_args)

if __name__ == "__main__":
    main()