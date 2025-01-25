import os
import sys

from tqdm import tqdm
from watermark_REPEAT_NO_CONTEXT_BERT import watermark_model as repeat_no_context_model_bert
import os
import json
# from models.Watermark.watermark_faster import watermark_model as model_faster
from watermark_REPEAT_NO_CONTEXT_M3E import watermark_model as repeat_no_context_model_m3e


def preprocess_txt(original_text):
    original_text = original_text.replace("\\n", " ")
    original_text = original_text.replace('''"''', " ")
    original_text = original_text.replace("'", " ")
    original_text = original_text.replace("[", " ")
    original_text = original_text.replace("]", " ")
    # original_text = original_text.replace(r"\s+", " ")
    return original_text


def repeat_watermark_detect(raw, model):
    is_watermark, p_value, n, ones, z_value = model.watermark_detector_fast_M3E(raw)
    confidence = (1 - p_value) * 100
    return z_value, p_value
    # return f"{confidence:.2f}%"

def rewrite_score(file_name, model):
    dataset = json.load(open(file_name, 'r'))

    output = []
    p_value = []
    error_message = []
    error_text = ''
    print(f"""
    ==============================
                重写数据
      file_name => {file_name}
      model => {model.embedding_model_path}
      tau_word => {tau_word_value}
    ==============================
    """)

    for i in tqdm(range(len(dataset))):
        try:
            rewrite_text = preprocess_txt(dataset[i]['rewrite_text'])
            z_score_rewrite, p_value = repeat_watermark_detect(rewrite_text, model)
            output.append({
                'original_text': dataset[i]['original_text'],
                'watermark_text': dataset[i]['watermark_text'],
                'rewrite_text': dataset[i]['rewrite_text'],
                "ori-fast-z-score": dataset[i]['ori-fast-z-score'],
                "water-fast-z-score": dataset[i]['water-fast-z-score'],
                'rewrite-fast-z-score': z_score_rewrite
            })
        except BaseException as e:
            error_message.append({
                'index': i,
                'content': dataset[i],
                'message': e
            })
            print(f'出现错误:{e}')
            error_text = f'出现错误'
    print('=== 完成 ===')
    print(f'==={error_text}===')

    if error_text =='':
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    file_name = 'result/M3E_REPEAT/gpt_rewrite/HC3-random-800.json'
    tau_word_value = 0.75
    lambda_value = 0.83
    # model = repeat_no_context_model_m3e(language="English", mode="embed", tau_word=tau_word_value, lamda=0.83)

    # model = repeat_no_context_model_bert(tau_word=tau_word_value)

    model = repeat_no_context_model_m3e(tau_word=tau_word_value)

    rewrite_score(file_name, model)

    # bloomz_file = 'result/M3E_REPEAT/gpt_rewrite/M4_arxiv_abstract/m4-bloomz-0.75-800.json'
    # chatGPT_file = 'result/M3E_REPEAT/gpt_rewrite/M4_arxiv_abstract/m4-chatGPT-0.75-800.json'
    # cohere_file = 'result/M3E_REPEAT/gpt_rewrite/M4_arxiv_abstract/m4-cohere-0.75-800.json'
    # davinci_file = 'result/M3E_REPEAT/gpt_rewrite/M4_arxiv_abstract/m4-davinci-0.75-800.json'
    # dolly_file = 'result/M3E_REPEAT/gpt_rewrite/M4_arxiv_abstract/m4-dolly-0.75-800.json'
    # flant5_file = 'result/M3E_REPEAT/gpt_rewrite/M4_arxiv_abstract/m4-flant5-0.75-800.json'

    # rewrite_score(bloomz_file, model)
    # rewrite_score(chatGPT_file, model)
    # rewrite_score(cohere_file, model)
    # rewrite_score(davinci_file, model)
    # rewrite_score(dolly_file, model)
    # rewrite_score(flant5_file, model)

    
    