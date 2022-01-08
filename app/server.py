from flask import Flask, request, jsonify
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

np.random.seed(42)
torch.manual_seed(42)

tok = GPT2Tokenizer.from_pretrained("model/")
model = GPT2LMHeadModel.from_pretrained("model/")

application = Flask(__name__)

@application.route("/aslince/", methods=['POST'])
def generate():
    text = request.json['text']
    text += "<end>"
    inpt = tok.encode(text, return_tensors="pt")
    out = model.generate(inpt,
            max_length=140,
            repetition_penalty=5.0,
            do_sample=True,
            top_k=5,
            top_p=0.95,
            temperature=1
        )
    gen = tok.decode(out[0])
    gen = gen.replace(text, '')
    if gen[0] == '\n':
        gen = gen[1:]
    gen = gen.replace('<line>', '\n')
    gen = gen.replace('<line', '\n')
    gen = gen.replace('line>', '\n')
    p = re.split('<break>|<break|break>', gen)
    resp = p[0]
    resp = resp.replace('>', '')
    resp = resp.replace('<', '')
    resp = re.sub(r'(\?|\)|!|0|\.)\1{3}(.)*(\n|$)', '', resp, flags=re.MULTILINE)

    sticker = ''
    if resp.lower().startswith('sticker'):
        m = re.search(r'\d+', resp)
        sticker = resp[m.start():m.end()]
        resp = ""

    if len(resp) > 140:
        resp = resp.split("\n")[0]

    return jsonify({'text': resp, 'sticker': sticker})

# if __name__ == "__main__":
#   answ = g.generate("привет")
#   print(answ)
