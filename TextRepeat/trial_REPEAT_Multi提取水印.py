from watermark_REPEAT_NO_CONTEXT_BERT import watermark_model as repeat_no_context_bert_model
import json
from tqdm import tqdm
from utils1.preprocess_text import preprocess_txt
import os

file_name = 'result/REPEAT/water/HC3-random-800.json'
result_file = 'result/REPEAT/water/HC3-random-800-multi.json'
model = repeat_no_context_bert_model(0.75)

dataset = json.load(open(file_name))



def repeat_watermark_detect(raw):
    is_watermark, p_value, n, ones, z_value = model.watermark_detect_bert_multi(raw)
    confidence = (1 - p_value) * 100
    return z_value,p_value
    # return f"{confidence:.2f}%"

print(f"""
    ==============================
      file_name => {file_name}
      result_file => {result_file}
      embedding_model_path => {model.embedding_model_path}
      tau_word => {model.tau_word}
    ==============================
    """)

output = []

for index in tqdm(range(len(dataset))):
    text = preprocess_txt(dataset[index]['original_text'])
    watermarked_text = preprocess_txt(dataset[index]['watermark_text'])
    fast_z_score_ori, p_value = repeat_watermark_detect(text)
    fast_z_score_water, p_value = repeat_watermark_detect(watermarked_text)
    output.append({
            'original_text': dataset[index]['original_text'],
            'watermark_text': dataset[index]['watermark_text'],
            "ori-fast-z-score": fast_z_score_ori,
            "water-fast-z-score": fast_z_score_water,
        })
    
os.makedirs(os.path.dirname(result_file), exist_ok=True)
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=4, ensure_ascii=False)