import torch
import lightning as L
from PIL import Image
import src.prompt_handler as prompt_handler

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle, Conversation

L.seed_everything(42)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.\n" +
    prompt_handler.DEFAULT_SYSTEM_PROMPT,
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


class LLaVA:

    def __init__(self,
                 model_path: str = "liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3",
                 load_in_4bit: bool = True):
        self.model_path = model_path
        self.load_in_4bit = load_in_4bit
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        disable_torch_init()
        self.model_name = get_model_name_from_path(self.model_path)
        args = load_pretrained_model(self.model_path, None, self.model_name, load_4bit=self.load_in_4bit)
        self.tokenizer, self.model, self.image_processor, self.context_len = args

    def create_conv(self):
        self.conv = conv_templates['llava_v1'].copy()

    def process_image(self, image):

        def expand2square(pil_img, background_color=(122, 116, 104)):
            width, height = pil_img.size
            image = pil_img
            if width > height:
                image = Image.new(pil_img.mode, (width, width), background_color)
                image.paste(pil_img, (0, (width - height) // 2))
            else:
                image = Image.new(pil_img.mode, (height, height), background_color)
                image.paste(pil_img, ((height - width) // 2, 0))

            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 800, 400
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
            return image

        image = expand2square(image)
        self.image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    def generate(self, pil_image: Image, hints: str, new_chat: bool = True):
        query = prompt_handler.get_prompt(hints)
        if new_chat:
            if self.model.config.mm_use_im_start_end:
                query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
            else:
                query = DEFAULT_IMAGE_TOKEN + '\n' + query
            self.process_image(pil_image)
            self.create_conv()

        self.conv.append_message(self.conv.roles[0], query)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(input_ids,
                                             images=self.image_tensor,
                                             do_sample=True,
                                             temperature=0.1,
                                             top_p=0.7,
                                             max_new_tokens=512,
                                             use_cache=True,
                                             stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        self.conv.messages[-1][-1] = outputs
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
