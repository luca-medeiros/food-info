import os
import warnings

import gradio as gr
import lightning as L
from lightning.app.components.serve import ServeGradio
from PIL import Image

from src.llava_wrapper import LLaVA
from src import prompt_handler

warnings.filterwarnings("ignore")


class LitGradio(ServeGradio):

    inputs = [
        gr.Image(type="filepath", label='Image'),
        gr.Textbox(label="Hints", placeholder="Menu name, brocolli 3 pieces, steak 150g..."),
    ]
    outputs = [
        gr.outputs.Textbox(label="Menu Name"),
        gr.outputs.Textbox(label="Description"),
        gr.outputs.JSON(label="Nutrients"),
    ]

    examples = [
        [
            os.path.join(os.path.dirname(__file__), "assets", "fruits.jpg"),
            None,
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets", "pasta.jpeg"),
            None,
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets", "pizza.jpeg"),
            None,
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets", "salad.jpeg"),
            None,
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets", "waffle.jpeg"),
            None,
        ],
    ]

    def __init__(self):
        super().__init__()
        self.ready = False

    def predict(self, image_path, hints):
        print("Start prediction")
        image_pil = Image.open(image_path).convert("RGB")
        output = self.model.generate(image_pil, hints)
        dict_out = prompt_handler.parse_dictionary_string(output)
        isfood = dict_out.get('imageofFood', 'False')
        if not isfood:
            return 'Not food', 'Not food', {}
        menu_name = dict_out.get('menuName', 'Not found')
        description = dict_out.get('description', 'Not found')
        nutrients = dict_out.get('nutrients', 'Not found')
        nutrients_dict = {}
        if type(nutrients) is list:
            for nutrient in nutrients:
                name = nutrient['nutrientName']
                value = nutrient['nutritionalValue']
                unit = nutrient['unit']
                nutrients_dict[name] = f"{value} {unit}"

        print('Done')

        return menu_name, description, nutrients_dict

    def build_model(self):
        model = LLaVA()
        self.ready = True
        return model


app = L.LightningApp(LitGradio())
