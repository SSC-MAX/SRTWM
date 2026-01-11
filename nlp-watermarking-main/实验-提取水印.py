import os.path
from functools import reduce
from itertools import product
import math
import re
import spacy
import string
import json
import os

import torch

from config import WatermarkArgs, GenericArgs, stop
from models.watermark import InfillModel
# from utils.logging import getLogger
from utils.metric import Metric
from torch import cuda
from utils.dataset_utils import preprocess2sentence
from tqdm import tqdm
from utils.misc import compute_ber
from scipy.stats import norm

# 禁用并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
def preprocess_txt(original_text):
    original_text = original_text.replace("\\n", " ")
    original_text = original_text.replace('''"''', " ")
    original_text = original_text.replace("'", " ")
    original_text = original_text.replace("[", " ")
    original_text = original_text.replace("]", " ")
    # original_text = original_text.replace(r"\s+", " ")
    return original_text

def detect_watermark(text, generic_args, dirname, spacy_tokenizer, device, model):
    
    bit_error_agg = {"sentence_err_cnt": 0, "sentence_cnt": 0}

    raw_text = text
    cover_texts = preprocess2sentence([raw_text], corpus_name="custom",
                                    start_sample_idx=0, cutoff_q=(0.0, 1.0), use_cache=False)


    DEBUG_MODE = generic_args.debug_mode
    dtype = "custom"
   
    start_sample_idx = 0
    num_sample = generic_args.num_sample

    
    if "trf" in generic_args.spacy_model:
        spacy.require_gpu()
    

    bit_count = 0
    word_count = 0
    upper_bound = 0
    candidate_kwd_cnt = 0
    kwd_match_cnt = 0
    mask_match_cnt = 0
    sample_cnt = 0
    one_cnt = 0
    zero_cnt = 0

    result_dir = os.path.join(dirname, "watermarked.txt")
    if not os.path.exists(result_dir):
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)

    kwd_match_cnt = 0
    midx_match_cnt = 0
    mword_match_cnt = 0
    num_mask_match_cnt = 0
    infill_match_cnt = 0

    n = 0
    ones = 0


    for c_idx, sentences in enumerate(cover_texts):
        corpus_level_watermarks = []
        for s_idx, sen in enumerate(sentences):
            sen = spacy_tokenizer(sen.text.strip())
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])
            keyword = all_keywords[0]
            ent_keyword = entity_keywords[0]

            agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sen, keyword, ent_keyword,
                                                                                  train_flag=False, embed_flag=True)
            
            wm_keys = model.tokenizer(" ".join([t.text for t in mask_word]), add_special_tokens=False)['input_ids']

            # 检查关键词和掩码位置匹配
            kwd_match_flag = set([x.text for x in keyword]) == set([x.text for x in all_keywords[0]])
            if kwd_match_flag:
                kwd_match_cnt += 1
        
            midx_match_flag = set(mask_idx) == set(mask_idx_pt)
            if midx_match_flag:
                midx_match_cnt += 1
        
            mword_match_flag = set([m.text for m in mask_word]) == set([m.text for m in mask_word])
            if mword_match_flag:
                mword_match_cnt += 1
        
            num_mask_match_flag = len(mask_idx) == len(mask_idx_pt)
            if num_mask_match_flag:
                num_mask_match_cnt += 1

            valid_watermarks = []
            valid_keys = []
            if len(agg_cwi) > 0:
                for cwi in product(*agg_cwi):
                    wm_text = [token.text_with_ws for token in sen]
                    for m_idx, c_id in zip(mask_idx, cwi):
                        wm_text[m_idx] = re.sub(r"\S+", model.tokenizer.decode(c_id), wm_text[m_idx])
                    valid_watermarks.append("".join(wm_text))
                    valid_keys.append(torch.stack(cwi).tolist())
                
        
            # 假设已知原始消息，计算比特错误率
            # 这里需要根据实际情况调整，例如从文件中读取原始消息
            original_message = '01'  # 示例原始消息
            # extracted_message = "010101"  # 示例提取的消息

            extracted_msg = []
            if len(valid_keys) > 1:
                try:
                    extracted_msg_decimal = valid_keys.index(wm_keys)
                except:
                    similarity = [len(set(wm_keys).intersection(x)) for x in valid_keys]
                    similar_key = max(zip(valid_keys, similarity), key=lambda x: x[1])[0]
                    extracted_msg_decimal = valid_keys.index(similar_key)

                num_digit = math.ceil(math.log2(len(valid_keys)))
                extracted_msg = format(extracted_msg_decimal, f"0{num_digit}b")
                extracted_msg = list(map(int, extracted_msg))

            print(f'===extracted_msg==>{type(extracted_msg[0])}, {extracted_msg}===')
            p=0.5
            n+=1
            alpha=0.05
            if original_message == extracted_msg[0]:
                ones += 1
            

    #         error_cnt, cnt = compute_ber(original_message, extracted_msg)
    #         bit_error_agg['sentence_err_cnt'] += error_cnt
    #         bit_error_agg['sentence_cnt'] += cnt

    # # 输出统计结果
    # print(f"BER: {bit_error_agg['sentence_err_cnt']}/{bit_error_agg['sentence_cnt']}=" 
    #   f"{bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']:.3f}")
    # print(f"Keyword match rate: {kwd_match_cnt}")
    # print(f"Mask index match rate: {midx_match_cnt}")
    # print(f"Mask word match rate: {mword_match_cnt}")
    # print(f"Number of mask match rate: {num_mask_match_cnt}")
    if n == 0:
        z = 0 
    else:
        z = (ones - p * n) / (n * p * (1 - p)) ** 0.5
    threshold = norm.ppf(1 - alpha, loc=0, scale=1)
    p_value = norm.sf(z)
    return z



if __name__ == '__main__':
    file_name = 'result/nlp/HC3-random-5.json'
    result_file = 'result/nlp/HC3-random-5-1.json'
    

    infill_parser = WatermarkArgs()
    infill_parser.add_argument("--custom_keywords", type=str)
    generic_parser = GenericArgs()
    infill_args, _ = infill_parser.parse_known_args()
    infill_args.mask_select_method = "grammar"
    infill_args.mask_order_by = "dep"
    infill_args.exclude_cc = True
    infill_args.topk = 2
    infill_args.dtype = None
    infill_args.model_name = '/root/autodl-tmp/bert-large-cased'
    # you can add your own entity / keyword that should NOT be masked.
    # This list will need to be saved when extracting
    infill_args.custom_keywords = ["watermarking", "watermark"]
    dirname = f"./results/ours/{123}/{456}"
    model = InfillModel(infill_args, dirname=dirname)

    generic_args, _ = generic_parser.parse_known_args()
    spacy_tokenizer = spacy.load(generic_args.spacy_model)

    # select device
    if torch.has_mps:
        device = "mps"
    elif cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    dataset = json.load(open(file_name))
    output = []

    detect_watermark(dataset[0]['watermark_text'], generic_args, dirname, spacy_tokenizer, device, model)
    print(f"original_message:{dataset[0]['original_message']}")

    # for index in tqdm(range(5)):
       
    #         original_text = preprocess_txt(dataset[index]['original_text'])
    #         watermark_text = preprocess_txt(dataset[index]['watermark_text'])
    #         ori_fast_z_score = detect_watermark(original_text, generic_args, dirname, spacy_tokenizer, device, model)
    #         water_fast_z_score = detect_watermark(watermark_text, generic_args, dirname, spacy_tokenizer, device, model)
    #         output.append({
    #             'original_text': original_text,
    #             'watermark_text': watermark_text,
    #             'ori-fast-z-score': ori_fast_z_score,
    #             'water-fast-z-score':water_fast_z_score
    #         })
        
    # os.makedirs(os.path.dirname(result_file), exist_ok=True)
    # with open(result_file, 'w', encoding='utf-8') as f:
    #     json.dump(output, f, indent=4, ensure_ascii=False)
        
    print('===水印完成===')