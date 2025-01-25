import os
import sys
import warnings

import json
from tqdm import tqdm
import pandas as pd
from utils1.preprocess_text import preprocess_txt
# from models.Watermark.watermark_faster import watermark_model as model_faster
from watermark_REPEAT_NO_CONTEXT_BERT import watermark_model as repeat_no_context_model_bert
import warnings

def watermark_embed_demo(raw):
    watermarked_text = model.embed(raw)
    return watermarked_text

def repeat_watermark_detect(raw):
    is_watermark, p_value, n, ones, z_value = model.watermark_detector_fast_bert(raw)
    confidence = (1 - p_value) * 100
    return z_value,p_value
    # return f"{confidence:.2f}%"

if __name__ == "__main__":
    # text = "Flocking is a type of coordinated group behavior that is exhibited by animals of various species, including birds, fish, and insects."
    
    file_name = 'nlp-watermarking-main/data/HC3/dataset-random-800.json'
    lambda_weight_list = ['1.0']

    for lambda_weight in lambda_weight_list:
        transform_model_path = f'TextRepeat/models/lambda_weight/transform_model-BERT-DISTANCE-CONDITION-{lambda_weight}.pth'
        result_file = f'result/REPEAT/lambda_weight/watermark/HC3-random-0.75-800-{lambda_weight}.json'
    
        tau_word_value = 0.75

        dataset = json.load(open(file_name, 'r'))
        model = repeat_no_context_model_bert(tau_word=tau_word_value, transform_model_path=transform_model_path)

        print(f"""
        ==============================
        file_name => {file_name}
        result_file => {result_file}
        embedding_model_path => {model.embedding_model_path}
        transform_model => {model.transform_path}
        tau_word => {model.tau_word}
        ==============================
        """)

        re_H = []
        re_C = []
        cnt = 0
        output = []
        for index in tqdm(range(len(dataset))):
            text = preprocess_txt(dataset[index])
            watermarked_text = watermark_embed_demo(text)
            fast_z_score_ori, p_value = repeat_watermark_detect(text)
            fast_z_score_water, p_value = repeat_watermark_detect(watermarked_text)

            output.append({
                'original_text': text,
                'watermark_text': watermarked_text,
                "ori-fast-z-score": fast_z_score_ori,
                "water-fast-z-score": fast_z_score_water,
            })
        
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

