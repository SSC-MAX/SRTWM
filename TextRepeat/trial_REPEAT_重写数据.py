import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(root_directory)

print('current directory:', current_directory)
print(root_directory)

from tqdm import tqdm
# from watermark_SENTENCE import watermark_model as sentence_model
from watermark_REPEAT_NO_CONTEXT_BERT import watermark_model as repeat_no_context_bert_model
# from utils1.preprocess_text import preprocess_txt
import os
import json


def preprocess_txt(original_text):
    original_text = original_text.replace("\\n", " ")
    original_text = original_text.replace('''"''', " ")
    original_text = original_text.replace("'", " ")
    original_text = original_text.replace("[", " ")
    original_text = original_text.replace("]", " ")
    # original_text = original_text.replace(r"\s+", " ")
    return original_text


def repeat_watermark_detect_bert(raw, model):
    is_watermark, p_value, n, ones, z_value = model.watermark_detector_fast_bert(raw)
    confidence = (1 - p_value) * 100
    return z_value, p_value
    # return f"{confidence:.2f}%"


if __name__ == '__main__':
    tau_word_value = 0.75
    lambda_value = 0.83

    dataset_name = ['0.8_0.9_1']

    for name in dataset_name:
        # file_name = '../../result/REPEAT/gpt/random-800.json'
        # file_name = f'result/REPEAT/lambda_weight/gpt_rewrite/HC3-random-0.75-800-{name}.json'
        # result_file = f'result/REPEAT/lambda_weight/gpt_rewrite/HC3-random-0.75-800-{name}.json'
        # transform_model_path = f'TextRepeat/models/lambda_weight/transform_model-BERT-DISTANCE-CONDITION-{name}.pth'

        file_name = f'result/REPEAT/gpt_rewrite/HC3-random-800_1.json'
        result_file = f'result/REPEAT/gpt_rewrite/HC3-random-800_1.json'
        # transform_model_path = f'TextRepeat/models/tau/transform_model-BERT-DISTANCE-CONDITION-0.75_0.8.pth'

        model = repeat_no_context_bert_model(tau_word=tau_word_value, transform_model_path='TextRepeat/models/transform_model-bert_large-DISTANCE-CONDITION.pth')
        # model = repeat_double_model(language="English", mode="embed", tau_word=tau_word_value, lamda=0.83)
        dataset = json.load(open(file_name, 'r'))

        output = []
        p_value = []
        error_message = []
        error_text = ''
        print(f"""
        ==============================
                    重写数据
        file_name => {file_name}
        result_file => {result_file}
        model => {model.embedding_model_path}
        tau_word => {tau_word_value}
        transform_model_path => {model.transform_path}
        ==============================
        """)
        
        try:
            for i in tqdm(range(len(dataset))):
                    rewrite_text = preprocess_txt(dataset[i]['translate_text'])
                    z_score_rewrite, p_value = repeat_watermark_detect_bert(rewrite_text, model)
                    output.append({
                        'original_text': dataset[i]['original_text'],
                        'watermark_text': dataset[i]['watermark_text'],
                        'rewrite_text': dataset[i]['translate_text'],
                        "ori-fast-z-score": dataset[i]['ori-fast-z-score'],
                        "water-fast-z-score": dataset[i]['water-fast-z-score'],
                        'rewrite-fast-z-score': z_score_rewrite
                    })
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=4, ensure_ascii=False)
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
    

    
