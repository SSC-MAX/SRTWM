import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import nltk
from nltk.tokenize import sent_tokenize
import json


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained(r'../code/model/google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text

if __name__ == "__main__":
    # Example usage
    # dp = DipperParaphraser(model="/work/kalpeshkrish_umass_edu/better-paraphrases/para-paraphrase-ctx-t5-xxl")
    dp = DipperParaphraser(model="../code/model/dipper-paraphraser-xxl")

    # prompt = "Tracy is a fox."
    # input_text = "It is quick and brown. It jumps over the lazy dog."

    with open("240327-ori-water-precise-50-yang.json", 'r', encoding="utf-8") as file:
        lines = json.load(file)
    output = []
    for data in lines:
        input_text = data["watermark_text"]

        lex_60 = dp.paraphrase(input_text, lex_diversity=60, order_diversity=0, do_sample=False, max_length=512)
        lex_60_order_20 = dp.paraphrase(input_text, lex_diversity=60, order_diversity=20, do_sample=False, max_length=512)

        output.append({
            "original_text": data["original_text"],
            "water_text": input_text,
            "ori-precise-z-score": data["ori-precise-z-score"],
            "water-precise-z-score": data["water-precise-z-score"],
            "ori-fast-z-score": data["ori-fast-z-score"],
            "water-fast-z-score":data["water-fast-z-score"],

            "lex_60": lex_60,
            "lex_60_order_20": lex_60_order_20,

        })


    with open("dipper-water-yang.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
