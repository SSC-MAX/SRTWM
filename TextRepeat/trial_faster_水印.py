import sys
import os

from tqdm import tqdm
from watermark_faster import watermark_model as faster_model
# from utils import test_utils, token_utils
import json
import os
from utils1.preprocess_text import preprocess_txt

def watermark_embed_demo(raw):
    watermarked_text = model.embed(raw)
    return watermarked_text

def fast_watermark_detect(raw, model):
    is_watermark, p_value, n, ones, z_value = model.watermark_detector_fast(raw)
    confidence = (1 - p_value) * 100
    return z_value,p_value
    # return f"{confidence:.2f}%"

if __name__ == '__main__':
    data_size = 800
    tau_word_value = 0.75

    file_name = f'result/FASTER/translate/HC3-random-800-e.json'
    result_file = f'result/FASTER/translate/HC3-random-800-e.json' # 添加水印后的数据

    dataset = json.load(open(file_name, 'r'))
    model = faster_model(tau_word=tau_word_value)
    output = []
    p_value = []
    error_text = ''

    print(f"""
    ==============================
            faster 重写数据
      file_name => {file_name}
      result_file => {result_file}
      tau_word => {tau_word_value}
    ==============================
    """)

    # 添加水印
    for i in tqdm(range(len(dataset))):
        translate_text = preprocess_txt(dataset[i]['translate_text'])
        z_score_trans, p_value = fast_watermark_detect(translate_text, model)
        output.append({
            'original_text': dataset[i]['original_text'],
            'watermark_text': dataset[i]['watermark_text'],
            'translate_text': dataset[i]['translate_text'],
            "ori-fast-z-score": dataset[i]['ori-fast-z-score'],
            "water-fast-z-score": dataset[i]['water-fast-z-score'],
            "trans-fast-z-score": z_score_trans
        })

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print('===水印添加完成===')