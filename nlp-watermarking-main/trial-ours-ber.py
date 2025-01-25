import os.path
from itertools import product
from datasets import load_dataset
import math
import random
import re
import spacy
import string
import sys

import torch
from tqdm.auto import tqdm

from config import WatermarkArgs, GenericArgs, stop
from models.watermark import InfillModel
from utils.dataset_utils import get_result_txt
from utils.logging import getLogger
# from utils.metric import Metric
from utils.misc import compute_ber
from utils.dataset_utils import preprocess2sentence, preprocess_txt

import json

import os

random.seed(1230)

infill_parser = WatermarkArgs()
generic_parser = GenericArgs()
infill_args, _ = infill_parser.parse_known_args()
generic_args, _ = generic_parser.parse_known_args()
infill_args.mask_select_method = "grammar"
infill_args.mask_order_by = "dep"
infill_args.exclude_cc = True
infill_args.topk = 2
infill_args.dtype = None
infill_args.model_name = '/root/autodl-tmp/bert-large-cased'
infill_args.custom_keywords = ["watermarking", "watermark"]
DEBUG_MODE = generic_args.debug_mode   # => False
dtype = generic_args.dtype   # => imdb

dirname = f"./results/ours/{dtype}/{generic_args.exp_name}"   # => .results/ours/imdb/tmp
start_sample_idx = 0
num_sample = generic_args.num_sample   # => 100

spacy_tokenizer = spacy.load(generic_args.spacy_model)
if "trf" in generic_args.spacy_model:
    spacy.require_gpu()
model = InfillModel(infill_args, dirname=dirname)

bit_count = 0
word_count = 0
upper_bound = 0
candidate_kwd_cnt = 0
kwd_match_cnt = 0
mask_match_cnt = 0
sample_cnt = 0
one_cnt = 0
zero_cnt = 0

device = torch.device("cuda")
# metric = Metric(device, **vars(generic_args))

if not os.path.exists(dirname):
    os.makedirs(dirname, exist_ok=True)

def get_list(string):
    output = []
    str_list = list(string.replace(" ", ""))
    for i in str_list:
        output.append(int(i))
    return output

def extract(dataset, prompt, result_file):
    #提取水印模式
    print(f'==={prompt}提取水印===')
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    z_score_result = []
    start_sample_idx = 0

    result_dir = os.path.join(dirname, "watermarked.txt")

    num_corrupted_sen = 0
    
    infill_match_cnt = 0

    num_mask_match_cnt = 0
    zero_cnt = 0
    one_cnt = 0
    kwd_match_cnt = 0
    midx_match_cnt = 0
    mword_match_cnt = 0

    output = []
    for index in tqdm(range(len(dataset))):  # 一段文本
        text_index = index  # 文本的索引
        sentence_length = len(dataset[index]) # 文本中句子的数目
        bit_error_agg = {"sentence_err_cnt": 0, "sentence_cnt": 0}
        text_result = []
        # 处理每个句子
        for data in dataset[index]:
            c_idx = data['c_idx']
            sen_idx = data['sen_idx']
            sub_idset = data['sub_idset']
            sub_idx = data['sub_idx']
            clean_wm_text = data['clean_wm_text']
            key = data['key']
            msg = data['msg']
            # print(f'==========\nc_idx:{c_idx}\nsen_idx:{sen_idx}\nsub_idset:{sub_idset}\nsub_idx:{sub_idx}\nclean_wm_text:{clean_wm_text}\ntext_type:{type(clean_wm_text)}\nkey:{key}\nmsg:{msg}\n==========')
            wm_texts = [clean_wm_text.strip()]
            for wm_text in wm_texts:
                num_corrupted_sen += 1
                sen = spacy_tokenizer(wm_text.strip())
                all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])
                # for sanity check, we use the uncorrupted watermarked texts
                sen_ = spacy_tokenizer(clean_wm_text.strip())
                all_keywords_, entity_keywords_ = model.keyword_module.extract_keyword([sen_])

                # extracting states for corrupted
                keyword = all_keywords[0]
                ent_keyword = entity_keywords[0]
                agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sen, keyword, ent_keyword,
                                                                                                  train_flag=False, embed_flag=True)
                wm_keys = model.tokenizer(" ".join([t.text for t in mask_word]), add_special_tokens=False)['input_ids']

                # extracting states for uncorrupted
                keyword_ = all_keywords_[0]
                ent_keyword_ = entity_keywords_[0]
                agg_cwi_, agg_probs_, tokenized_pt_, (mask_idx_pt_, mask_idx_, mask_word_) = model.run_iter(sen_, keyword_, ent_keyword_,
                                                                                                        train_flag=False, embed_flag=True)
                kwd_match_flag = set([x.text for x in keyword]) == set([x.text for x in keyword_])
                if kwd_match_flag:
                    kwd_match_cnt += 1

                midx_match_flag = set(mask_idx) == set(mask_idx_)
                if midx_match_flag:
                    midx_match_cnt += 1

                mword_match_flag = set([m.text for m in mask_word]) == set([m.text for m in mask_word_])
                if mword_match_flag:
                    mword_match_cnt += 1

                num_mask_match_flag = len(mask_idx) == len(mask_idx_)
                if num_mask_match_flag:
                    num_mask_match_cnt += 1

                infill_match_list = []
                if len(agg_cwi) == len(agg_cwi_):
                    for a, b in zip(agg_cwi, agg_cwi_):
                        if len(a) == len(b):
                            infill_match_flag = (a == b).all()
                        else:
                            infill_match_flag = False
                        infill_match_list.append(infill_match_flag)
                else:
                    infill_match_list.append(False)
                if all(infill_match_list):
                    infill_match_cnt += 1

                valid_watermarks = []
                valid_keys = []
                tokenized_text = [token.text_with_ws for token in sen]

                if len(agg_cwi) > 0:
                    for cwi in product(*agg_cwi):
                        wm_text = tokenized_text.copy()
                        for m_idx, c_id in zip(mask_idx, cwi):
                            wm_text[m_idx] = re.sub(r"\S+", model.tokenizer.decode(c_id), wm_text[m_idx])

                        wm_tokenized = spacy_tokenizer("".join(wm_text).strip())
                        # extract keyword of watermark
                        wm_keywords, wm_ent_keywords = model.keyword_module.extract_keyword([wm_tokenized])
                        wm_kwd = wm_keywords[0]
                        wm_ent_kwd = wm_ent_keywords[0]
                        wm_mask_idx, wm_mask = model.mask_selector.return_mask(wm_tokenized, wm_kwd, wm_ent_kwd)

                        # checking whether the watermark can be embedded
                        mask_match_flag = len(wm_mask) > 0 and set(wm_mask_idx) == set(mask_idx)
                        if mask_match_flag:
                            valid_watermarks.append(wm_tokenized.text)
                            valid_keys.append(torch.stack(cwi).tolist())

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

                one_cnt += sum(msg)
                zero_cnt += len(msg) - sum(msg)

                # print(f'======\ntype(msg)=>{type(msg)}\nmsg=>{msg}\ntype(extracted_msg)=>{type(extracted_msg)}\nextracted_msg=>{extracted_msg}')

                # error_cnt, cnt = compute_ber(msg, extracted_msg)
                # bit_error_agg['sentence_err_cnt'] = bit_error_agg.get('sentence_err_cnt', 0) + error_cnt
                # bit_error_agg['sentence_cnt'] = bit_error_agg.get('sentence_cnt', 0) + cnt
                # if bit_error_agg['sentence_cnt']:
                #     ber = bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']
                #     error = 1
                # else:
                #     ber = 0
                #     error = 0
                error_ber = 0
                if msg != extracted_msg:
                    error_ber += 1
                
                text_result.append({
                    'c_idx': data['c_idx'],
                   'sen_idx' : data['sen_idx'],
                    'sub_idset' : data['sub_idset'],
                    'sub_idx' : data['sub_idx'],
                    'clean_wm_text' : data['clean_wm_text'],
                    'key' : data['key'],
                    'error_bit': error_ber,
                    'msg': msg,
                    'extracted_msg': extracted_msg,
                })

        output.append(text_result)

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

    print(f'==={prompt}提取完成===')

if __name__ == '__main__':   # 嵌入水印模式
    file_name = 'nlp-watermarking-main/data/HC3/dataset-random-800.json'
    watermark_result = 'nlp-watermarking-main/result/nlp/HC3/watermark/HC3-random-800.json'
    result_file = 'nlp-watermarking-main/result/nlp/ber/HC3-random-800-count.json'

    print(f"""
    ==============================
      file_name => {file_name}
      watermark_result => {watermark_result}
      result_file => {result_file}
    ==============================
    """)

    watermark_text_output = json.load(open(watermark_result))

    water_fast_z_score = extract(watermark_text_output, '水印文本', result_file)

      
      






