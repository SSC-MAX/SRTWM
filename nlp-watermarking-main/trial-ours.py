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
from utils.metric import Metric
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
metric = Metric(device, **vars(generic_args))

if not os.path.exists(dirname):
    os.makedirs(dirname, exist_ok=True)

def get_list(string):
    output = []
    str_list = list(string.replace(" ", ""))
    for i in str_list:
        output.append(int(i))
    return output

def extract(dataset, prompt):
    #提取水印模式
    print(f'==={prompt}提取水印===')
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

    for index in tqdm(range(len(dataset))):  # 一段文本
        text_index = index  # 文本的索引
        sentence_length = len(dataset[index]) # 文本中句子的数目
        bit_error_agg = {"sentence_err_cnt": 0, "sentence_cnt": 0}
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

                error_cnt, cnt = compute_ber(msg, extracted_msg)
                bit_error_agg['sentence_err_cnt'] = bit_error_agg.get('sentence_err_cnt', 0) + error_cnt
                bit_error_agg['sentence_cnt'] = bit_error_agg.get('sentence_cnt', 0) + cnt

        # 做z-test
        # ber = bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']
        ones = bit_error_agg['sentence_cnt'] - bit_error_agg['sentence_err_cnt']
        n = bit_error_agg['sentence_cnt']
        p = 0.5  # Null hypothesis: perfect extraction (BER = 0)
        # se = math.sqrt((ber * (1 - ber)) / n)
        if n == 0:
            z_score = 0 
        else:
            z_score = (ones - p * n) / (n * p * (1 - p)) ** 0.5
        # print(f"==========\nSentence BER: {bit_error_agg['sentence_err_cnt']}/{bit_error_agg['sentence_cnt']}="f"{ber}\n==========")
        z_score_result.append({
            'text-index': text_index,
            'sentence-length': sentence_length,
            'z-score': z_score
        })
        # print(f'======\n{z_score_result}\n======')
    return z_score_result

if __name__ == '__main__':   # 嵌入水印模式
    file_name = 'nlp-watermarking-main/data/HC3/dataset-random-800.json'
    watermark_result = 'nlp-watermarking-main/result/nlp/HC3/watermark/HC3-random-800.json'
    original_result = 'nlp-watermarking-main/result/nlp/HC3/original/HC3-random-800.json'
    result_file = 'nlp-watermarking-main/result/nlp/HC3/z_score/HC3-random-800-123.json'

    print(f"""
    ==============================
      file_name => {file_name}
      watermark_result => {watermark_result}
      original_result => {original_result}
      result_file => {result_file}
    ==============================
    """)

    dataset = json.load(open(file_name))

    cover_texts = preprocess2sentence(preprocess_txt(dataset), corpus_name="custom", start_sample_idx=0, cutoff_q=(0.0, 1.0), use_cache=False)

    print('===嵌入水印===')

    watermark_text_output = []
    original_text_output = []
    

    # 创建结果文件
    result_dir = os.path.join(dirname, "watermarked.txt")
    if not os.path.exists(result_dir):
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)

    progress_bar = tqdm(range(len(cover_texts)))
    wr = open(result_dir, "w")
    for c_idx, sentences in enumerate(cover_texts):
        text_result = []    # 每段文本的结果，存储每段文本中每个句子添加水印后的版本与嵌入的比特信息
        original_text_result = []
        for s_idx, sen in enumerate(sentences):
            sen = spacy_tokenizer(sen.text.strip())     # 将句子划分成单词
            # all_keywords: 全体关键词； entity_keyword: 实体关键词
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])   # 提取关键词
            keyword = all_keywords[0]
            ent_keyword = entity_keywords[0]
            agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sen, keyword, ent_keyword,
                                                                                                  train_flag=False, embed_flag=True)
            # check if keyword & mask_indices matches
            valid_watermarks = []
            candidate_kwd_cnt = 0
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

                    kwd_match_flag = set([x.text for x in wm_kwd]) == set([x.text for x in keyword])
                    if kwd_match_flag:
                        kwd_match_cnt += 1

                    # checking whether the watermark can be embedded without the assumption of corruption
                    mask_match_flag = len(wm_mask) and set(wm_mask_idx) == set(mask_idx)
                    if mask_match_flag:
                        valid_watermarks.append(wm_tokenized.text)
                        mask_match_cnt += 1

                    sample_cnt += 1
                    candidate_kwd_cnt += 1

            punct_removed = sen.text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
            word_count += len([i for i in punct_removed.split(" ") if i not in stop])
            if len(valid_watermarks) > 1:
                bit_count += math.log2(len(valid_watermarks))
                random_msg_decimal = random.choice(range(len(valid_watermarks)))
                num_digit = math.ceil(math.log2(len(valid_watermarks)))
                random_msg_binary = format(random_msg_decimal, f"0{num_digit}b")

                wm_text = valid_watermarks[random_msg_decimal]
                watermarked_text = "".join(wm_text) if len(mask_idx) > 0 else ""
                message_str = list(random_msg_binary)
                one_cnt += len([i for i in message_str if i =="1"])
                zero_cnt += len([i for i in message_str if i =="0"])

                keys = []
                wm_tokenized = spacy_tokenizer(wm_text)
                for m_idx in mask_idx:
                    keys.append(wm_tokenized[m_idx].text)
                keys_str = ", ".join(keys)
                message_str = ' '.join(message_str) if len(message_str) else ""
                # ''.join(wm_text): 水印文本
                # keys_str: 替换的词
                # message_str: 嵌入的比特信息
                wr.write(f"{c_idx}\t{s_idx}\t \t \t"
                         f"{''.join(wm_text)}\t{keys_str}\t{message_str}\n")  
                watermarksss_text = ''.join(wm_text)
                keys = []
                keys.append(keys_str)
                msg = get_list(message_str)
                # msg.append(int(message_str))
                text_result.append({
                    'c_idx': c_idx,
                    'sen_idx': s_idx,
                    'sub_idset':[],
                    'sub_idx':[],
                    'clean_wm_text':watermarksss_text,
                    'key': keys_str,
                    'msg': msg
                })
                original_text = ''.join(tokenized_text)
                original_text_result.append({
                    'c_idx': c_idx,
                    'sen_idx': s_idx,
                    'sub_idset':[],
                    'sub_idx':[],
                    'clean_wm_text':original_text,
                    'key': keys_str,
                    'msg': msg
                })
            else:
                # 无可添加水印，直接将原始文本作为水印
                original_text = ''.join(tokenized_text)
                wr.write(f"{c_idx}\t{s_idx}\t \t \t"
                         f"{original_text}\t \t \n")
                text_result.append({
                    'c_idx': c_idx,
                    'sen_idx': s_idx,
                    'sub_idset':[],
                    'sub_idx':[],
                    'clean_wm_text': original_text,
                    'key': '',
                    'msg': []
                })
                original_text_result.append({
                    'c_idx': c_idx,
                    'sen_idx': s_idx,
                    'sub_idset':[],
                    'sub_idx':[],
                    'clean_wm_text':original_text,
                    'key': '',
                    'msg': []
                })

            # if candidate_kwd_cnt > 0:
            #     upper_bound += math.log2(candidate_kwd_cnt)
        progress_bar.update(1)
        watermark_text_output.append(text_result)
        original_text_output.append(original_text_result)
    
    os.makedirs(os.path.dirname(watermark_result), exist_ok=True)
    with open(watermark_result, 'w', encoding='utf-8') as f:
        json.dump(watermark_text_output, f, indent=4, ensure_ascii=False)

    os.makedirs(os.path.dirname(original_result), exist_ok=True)
    with open(original_result, 'w', encoding='utf-8') as f:
        json.dump(original_text_output, f, indent=4, ensure_ascii=False)

    original_text_output = json.load(open(original_result))
    watermark_text_output = json.load(open(watermark_result))

    ori_fast_z_score = extract(original_text_output, '原始文本')
    water_fast_z_score = extract(watermark_text_output, '水印文本')

    output_result = []

    for index in range(len(ori_fast_z_score)):
        output_result.append({
            'text-index': ori_fast_z_score[index]['text-index'],
            'sentence-length': ori_fast_z_score[index]['sentence-length'],
            'ori-fast-z-score': ori_fast_z_score[index]['z-score'],
            'water-fast-z-score': water_fast_z_score[index]['z-score']

        })

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output_result, f, indent=4, ensure_ascii=False)

      
      






