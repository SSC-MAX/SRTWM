from watermark_REPEAT_NO_CONTEXT_BERT_BER import watermark_model as repeat_no_context_bert_model_ber
import json
from tqdm import tqdm
from utils1.preprocess_text import preprocess_txt
import os
import random

def repeat_watermark_detect(raw):
    is_watermark, p_value, n, ones, z_value = model.watermark_detect_bert_multi(raw)
    confidence = (1 - p_value) * 100
    return z_value,p_value
    # return f"{confidence:.2f}%"



if __name__ == '__main__':
    file_name = 'result/REPEAT-MULTI/HC3/dataset-HC3-random-800.json'
    result_file = 'result/REPEAT-MULTI/HC3/ber/HC3-random-800-multi-random.json'

    model = repeat_no_context_bert_model_ber(0.75)
    dataset = json.load(open(file_name))[:100]

    random.seed(1230)
    embedding_bit = []
    for i in range(3):
        embedding_bit.append(random.choice([0, 1]))

    print(f"""
    ==============================
      file_name => {file_name}
      result_file => {result_file}
      embedding_model_path => {model.embedding_model_path}
      tau_word => {model.tau_word}
      embedding_bit => {embedding_bit}
    ==============================
    """)

    output = []

    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    for index in tqdm(range(len(dataset))):
        text = dataset[index]
        sentence_output = []
        for sentence in text:
            original_text = preprocess_txt(sentence['text'])
            original_message = embedding_bit
            watermark_text = model.wm_embed(original_text, original_message)
            error_count, extracted_message = model.watermark_detect_bert_multi(watermark_text, original_message)
            sentence_output.append({
                'original_text': original_text,
                'watermark_text': watermark_text,
                'error_bit': error_count,
                'original_message': original_message,
                'extracted_message': extracted_message
            })
        output.append(sentence_output)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        

    # for index in tqdm(range(len(dataset))):
    #     text = preprocess_txt(dataset[index]['original_text'])
    #     watermarked_text = preprocess_txt(dataset[index]['watermark_text'])
    #     fast_z_score_ori, p_value = repeat_watermark_detect(text)
    #     fast_z_score_water, p_value = repeat_watermark_detect(watermarked_text)
    #     output.append({
    #         'original_text': dataset[index]['original_text'],
    #         'watermark_text': dataset[index]['watermark_text'],
    #         "ori-fast-z-score": fast_z_score_ori,
    #         "water-fast-z-score": fast_z_score_water,
    #     })
    
    
   
