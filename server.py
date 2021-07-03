import logging
from tokenize import generate_tokens

import numpy as np
import torch
import json

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

import tornado.ioloop
import tornado.web


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(n_gpu, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

device = torch.device("cpu")

class Gen:
    def __init__(self) -> None:
        config = json.load(open("generation_config.json"))
        model_type = config['model_type']
        model_path = config['model_name_or_path']
        
        self.length = config['length']
        self.temperature = config['temperature']
        self.k = config['k']
        self.p = config['p']
        self.repetition_penalty = config['repetition_penalty']
        self.stop_token = config['stop_token']

        set_seed(n_gpu=0, seed=config['seed'])
        
        # Initialize the model and tokenizer
        try:
            model_class, tokenizer_class = MODEL_CLASSES[model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        self.tokenizer = tokenizer_class.from_pretrained(model_path)
        self.model = model_class.from_pretrained(model_path)
        self.model.to(device)
        
        config['length'] = adjust_length_to_model(self.length, max_sequence_length=self.model.config.max_position_embeddings)

        logger.info(device)

    def generate(self, input_text) -> str:
        # prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
        prompt_text = input_text.replace('\n', ' ')

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=self.length + len(encoded_prompt[0]),
            temperature=self.temperature,
            top_k=self.k,
            top_p=self.p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            num_return_sequences=1,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        generated_sequence = output_sequences[0].tolist()

        # Decode text
        text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        stop_token = self.stop_token
        text = text[:text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        
        return generated_sequences[0].split('\n')[1]
    

g = Gen()

class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", 'application/json')

    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        text = g.generate(data['text'])
        r = tornado.escape.json_encode({'text': text})
        print(text)
        self.write(r)

def make_app():
    return tornado.web.Application([
        (r"/generate", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()

# if __name__ == "__main__":
#    answ = g.generate("привет")
#    print(answ)
