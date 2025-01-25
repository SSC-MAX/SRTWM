from watermark_REPEAT_NO_CONTEXT_BERT import watermark_model as repeat_no_context_bert_model
import json
import os
from utils1.preprocess_text import preprocess_txt


def count_bit(text, result_file_1, result_file_0, model):
    sentences = model.sent_tokenize(text)
    sentences = [s for s in sentences if s.strip()]
    bit1 = []
    bit0 = []
    sentences = sentences[:3]
    print(len(sentences))
    for sent in sentences:
        tokens = model.tokenizer.tokenize(sent)
        greenlist_ids = model.get_greenlist_ids_bert(sent)
        for masked_token_index in range(0, len(tokens)):
            # print(f'==={masked_token_index}===')
            if not model.pos_filter(tokens, masked_token_index, sent):
                continue
            binary_encoding = 1 if model.tokenizer.convert_tokens_to_ids(
                    tokens[masked_token_index]) in greenlist_ids else 0
            if binary_encoding == 1:
                bit1.append({
                    'word': tokens[masked_token_index],
                    'bit':1
                })
            else:
                bit0.append({
                    'word': tokens[masked_token_index],
                    'bit':0
                })
    os.makedirs(os.path.dirname(result_file_1), exist_ok=True)
    with open(result_file_1, 'w', encoding='utf-8') as f:
        json.dump(bit1, f, indent=4, ensure_ascii=False)
    
    os.makedirs(os.path.dirname(result_file_0), exist_ok=True)
    with open(result_file_0, 'w', encoding='utf-8') as f:
        json.dump(bit0, f, indent=4, ensure_ascii=False)
    
    return ''.join(sentences)
        
if __name__ == '__main__':
    dataset = json.load(open('result/REPEAT/water/HC3-random-800.json'))
    model = repeat_no_context_bert_model()

    index = 34
    original_text = preprocess_txt(dataset[index]['original_text'])
    watermark_text = preprocess_txt(dataset[index]['watermark_text'])

    original_text = count_bit(original_text, 
              f'result/REPEAT/case_study/index{index}/HC3-original-1.json', 
              f'result/REPEAT/case_study/index{index}/HC3-original-0.json', 
              model)
    
    watermark_text = count_bit(watermark_text, 
              f'result/REPEAT/case_study/index{index}/HC3-watermark-1.json', 
              f'result/REPEAT/case_study/index{index}/HC3-watermark-0.json', 
              model)
    
    print(f'==========\n{original_text}\n==========\n{watermark_text}\n==========')
    



    
        

