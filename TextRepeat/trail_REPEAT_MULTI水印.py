import os
import sys
import warnings

import json
from tqdm import tqdm
import pandas as pd
from utils1.preprocess_text import preprocess_txt
# from models.Watermark.watermark_faster import watermark_model as model_faster
from watermark_REPEAT_NO_CONTEXT_BERT import watermark_model as repeat_no_context_model_bert
from watermark_REPEAT_NO_CONTEXT_BERT_MULTI import watermark_model as repeat_no_context_model_bert_multi
import warnings

if __name__ == "__main__":
    # text = "Flocking is a type of coordinated group behavior that is exhibited by animals of various species, including birds, fish, and insects."
    
    file_name = 'nlp-watermarking-main/data/HC3/dataset-random-800.json'
    result_file = "result/REPEAT-MULTI/water/HC3-random-800-multi.json"
    
    tau_word_value = 0.75

    dataset = json.load(open(file_name, 'r'))
    model = repeat_no_context_model_bert_multi(tau_word=tau_word_value)

    print(f"""
    ==============================
      file_name => {file_name}
      result_file => {result_file}
      embedding_model_path => {model.embedding_model_path}
      tau_word => {model.tau_word}
    ==============================
    """)

    re_H = []
    re_C = []
    cnt = 0
    output = []

    print(model.embedding_bit)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    for index in tqdm(range(len(dataset))):
        text = preprocess_txt(dataset[index])
        watermarked_text, embedding_bit = model.embed(text)
        # ber_ori, p_value = model.watermark_detect_bert_multi(text)
        ber_water = model.watermark_detect_bert_multi(watermarked_text)

        output.append({
            'original_text': text,
            'watermark_text': watermarked_text,
            "water-ber": ber_water,
            "embedding_bit": embedding_bit
        })
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
    
    

