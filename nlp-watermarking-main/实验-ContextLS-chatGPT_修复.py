# Key functions written by Wonhyk Ahn (whahnize@gmail.com)
# Refactoring and edits by KiYoon

from functools import reduce
import math
import os
import string
import random

from datasets import load_dataset
from tqdm import tqdm
import torch

from config import GenericArgs
from utils.misc import compute_ber, riskset, stop
from utils.contextls_utils import synchronicity_test, substitutability_test, tokenizer, riskset, stop
from utils.logging import getLogger
from utils.dataset_utils import preprocess_txt, preprocess2sentence, get_result_txt, get_dataset
from utils.metric import Metric

import json

random.seed(1230)

def get_list(string):
    output = []
    string = string.replace(",", "")
    str_list = list(string.replace(" ", ""))
    for i in str_list:
        output.append(int(i))
    return output

def main(cover_text, f, extracting=False):
    """
    cover_text: 待处理的单个句子
    """
    encoded_text = tokenizer(cover_text, add_special_tokens=False, truncation=True,
                             max_length=tokenizer.model_max_length // 2 - 2)    # 编码
    word_ids = encoded_text._encodings[0].word_ids   # 获取句子中每个单词对应的id
    watermarking_wordset = [None] * len(encoded_text['input_ids'])   # 水印词集
    substituted_idset = []   # 被替换的词的id集合
    substituted_indices = []   # 被替换的索引
    watermarked_text = []   # 水印文本
    message = []   # 嵌入的信息

    latest_embed_index = -1
    index = 1
    while index < len(encoded_text['input_ids']) - f:   # 从第二个词开始处理，直到划定的f为止
        text = tokenizer.decode(encoded_text['input_ids'][index])  # 还原当前处理的词
        watermarked_text.append(text)   # 

        if text in riskset:   # 若该词在风险集中，就不处理该词
            watermarking_wordset[index] = 'riskset'
            index = index + 1
            continue

        valid_indx = [t == index for t in word_ids]
        if sum(valid_indx) >= 2:  # skip this word; subword
            watermarking_wordset[index] = 'subword'
            index = index + 1
            continue

        local_context = encoded_text['input_ids'][:index + 2]  # 获取当前词的局部上下文
        is_synch, words = synchronicity_test(index, local_context)   # 做同步性测试

        if not is_synch:  # 不具备同步性，跳过该词
            watermarking_wordset[index] = 'syn'
            index = index + 1
            continue

        if index - latest_embed_index != f + 1:   # 当前词与上一个词之间的距离不超过f，做可替换性测试
            if not substitutability_test(local_context[:-1], index,
                                         words):  # skip this word if substitutability test fails
                watermarking_wordset[index] = 'sub'    # 没通过可替换性测试，跳过该词
                index = index + 1
                continue

        watermarking_wordset[index] = tokenizer.decode(words)
        words_decoded = tokenizer.decode(words).split(" ")
        # sort token_id alphabetically
        words = [w for w, wd in sorted(zip(words, words_decoded), key=lambda pair: pair[1])]
        words_decoded.sort()

        # extraction
        if extracting:
            extracted_msg = words_decoded.index(text)
            message.append(extracted_msg)

        # embedding
        else:
            random_msg = random.choice([0,1])
            word_chosen_by_msg = words[random_msg]
            encoded_text['input_ids'][index] = word_chosen_by_msg
            message.append(random_msg)


        substituted_idset.append(words)
        substituted_indices.append(index)
        latest_embed_index = index
        index = index + f + 1

    return substituted_idset, substituted_indices, watermarking_wordset, encoded_text['input_ids'], message

## 提取水印
def extract_watermark(dataset, z_score_file, f=1):
    z_score_result = []

    corrupted_watermarked = []

    num_corrupted_sen = 0
    sample_level_bit = {'gt':[], 'extracted':[]}
    bit_error_agg = {}
    midx_match_cnt = 0
    mword_match_cnt = 0
    infill_match_cnt = 0
    prev_c_idx = 0
    os.makedirs(os.path.dirname(z_score_file), exist_ok=True)
        
    for index in tqdm(range(len(dataset))):
        text_index = index
        sentence_length = len(dataset[index])
        bit_error_agg = {"sentence_err_cnt": 0, "sentence_cnt": 0}
        for idx in range(len(dataset[index])):  # 处理每个句子
            try:
                data = dataset[index][idx]
                c_idx = data['c_idx']
                sen_idx = data['sen_idx']
                sub_idset = data['s_idset_str']
                sub_idx = data['s_indices_str']
                clean_wm_sen = data['clean_wm_text']
                key = data['keys_str']
                msg = data['message_str']

                if prev_c_idx != c_idx:
                    error_cnt, cnt = compute_ber(sample_level_bit['extracted'], sample_level_bit['gt'])
                    bit_error_agg['sample_err_cnt'] = bit_error_agg.get('sample_err_cnt', 0) + error_cnt
                    bit_error_agg['sample_cnt'] = bit_error_agg.get('sample_cnt', 0) + cnt
                    sample_level_bit = {'gt': [], 'extracted': []}
                    prev_c_idx = c_idx

                if len(corrupted_watermarked) > 0 and len(corrupted_watermarked) <= idx:
                    break

                clean_encoded = tokenizer(clean_wm_sen.strip(), add_special_tokens=False, truncation=True,
                                        max_length=tokenizer.model_max_length// 2 - 2)['input_ids']
            
                wm_texts = [clean_wm_sen.strip()]
                for wm_text in wm_texts:
                    wm_text = wm_text.strip()

                    num_corrupted_sen += 1
                    extracted_idset, extracted_indices, watermarking_wordset, encoded_text, extracted_msg = \
                        main(wm_text, f, extracting=True)
                    extracted_key = [tokenizer.decode(s_id) for s_id in extracted_idset]

                    midx_match_flag = set(extracted_indices) == set(sub_idx)
                    if midx_match_flag:
                        midx_match_cnt += 1

                    mword_match_flag = set([encoded_text[idx] for idx in extracted_indices]) == \
                                    set([clean_encoded[idx] for idx in sub_idx])
                    if mword_match_flag:
                        mword_match_cnt += 1

                    infill_match_list = []
                    if len(sub_idset) == len(extracted_idset):
                        for a, b in zip(sub_idset, extracted_idset):
                            infill_match_flag = a==b
                            infill_match_list.append(infill_match_flag)
                    else:
                        infill_match_list.append(False)
                    if all(infill_match_list):
                        infill_match_cnt += 1

                    sample_level_bit['extracted'].extend(extracted_msg)
                    sample_level_bit['gt'].extend(msg)
                    error_cnt, cnt = compute_ber(msg, extracted_msg)
                    bit_error_agg['sentence_err_cnt'] = bit_error_agg.get('sentence_err_cnt', 0) + error_cnt
                    bit_error_agg['sentence_cnt'] = bit_error_agg.get('sentence_cnt', 0) + cnt

                    match_flag = error_cnt == 0
            except BaseException as e:
                print(f'===e => {e}===')

        # 做z-test
        ones = bit_error_agg['sentence_cnt'] - bit_error_agg['sentence_err_cnt']
        n = bit_error_agg['sentence_cnt']
        p = 0.5  # Null hypothesis: perfect extraction (BER = 0)
        if n == 0:
            z_score = 0 
        else:
            z_score = (ones - p * n) / (n * p * (1 - p)) ** 0.5
        z_score_result.append({
            'text-index': text_index,
            'sentence-length': sentence_length,
            'z-score': z_score
        })
        with open(z_score_file, 'w', encoding='utf-8') as store:
            json.dump(z_score_result, store, indent=4, ensure_ascii=False)
    return z_score_result

# if __name__ == '__main__':
def embed(file_name, original_file, watermark_file, name):
    f = 1
    start_sample_idx = 0

    # corpus: 数据集
    dataset = json.load(open(file_name))
    cover_texts = preprocess2sentence(preprocess_txt(dataset), corpus_name="custom", start_sample_idx=0, cutoff_q=(0.0, 1.0), use_cache=False)

    bit_count = 0
    word_count = 0

    # 嵌入水印
    original_output = []
    watermark_output = []
    for c_idx, sentences in enumerate(tqdm(cover_texts)):  # setences: 每一段文本
        original_result = []
        watermark_result = []
        for sen_idx, sen in enumerate(sentences):    # 每一个句子
            sen_text = sen.text    
            substituted_idset, substituted_indices, watermarking_wordset, encoded_text, message = main(sen_text, f)
            punct_removed = sen_text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
            word_count += len([i for i in punct_removed.split(" ") if i not in stop])
            num_cases = 1
            for sid in substituted_idset:
                num_cases *= len(sid)
            bit_count += math.log2(num_cases)
            s_idset_str = ""
            for s_id in substituted_idset:
                s_idset_str += " ".join(str(x) for x in s_id) + ","
            s_indices_str = " ".join(str(x) for x in substituted_indices)
            message_str = [str(m) for m in message]
            message_str = " ".join(message_str) if len(message_str) else ""
            watermarked_text = tokenizer.decode(encoded_text)
            keys = [tokenizer.decode(s_id) for s_id in substituted_idset]
            keys_str = ", ".join(keys)
            # wr.write(f"{c_idx}\t{sen_idx}\t{s_idset_str}\t{s_indices_str}\t"
            #              f"{watermarked_text}\t{keys_str}\t{message_str}\n")
            message_str = get_list(message_str)
            s_idset_str = get_list(s_idset_str)
            s_indices_str = get_list(s_indices_str)
            watermark_result.append({
                'c_idx': c_idx,
                'sen_idx': sen_idx,
                's_idset_str': s_idset_str,
                's_indices_str': s_indices_str,
                'clean_wm_text': watermarked_text,
                'keys_str': keys_str,
                'message_str': message_str
            })
            original_result.append({
                'c_idx': c_idx,
                'sen_idx': sen_idx,
                's_idset_str': s_idset_str,
                's_indices_str': s_indices_str,
                'clean_wm_text': sen.text,
                'keys_str': keys_str,
                'message_str': message_str
            })
        original_output.append(original_result)
        watermark_output.append(watermark_result)

    # print(original_output[0])

    os.makedirs(os.path.dirname(original_file), exist_ok=True)
    with open(original_file, 'w', encoding='utf-8') as f:
        json.dump(original_output, f, indent=4, ensure_ascii=False)

    os.makedirs(os.path.dirname(watermark_file), exist_ok=True)
    with open(watermark_file, 'w', encoding='utf-8') as f:
        json.dump(watermark_output, f, indent=4, ensure_ascii=False)

    return original_output, watermark_output

if __name__ == '__main__':
    parser = GenericArgs()
    args = parser.parse_args()

    dataset_name = ['chatGPT']

    for name in dataset_name:
        file_name = f'nlp-watermarking-main/M4/arxiv/abstract/dataset-M4-random-{name}-800.json'
        original_file = f'nlp-watermarking-main/result/ContextLS/M4/original/{name}-random-800.json'
        watermark_file = f'nlp-watermarking-main/result/ContextLS/M4/watermark/{name}-random-800.json'
        original_z_score_files = f'nlp-watermarking-main/result/ContextLS/M4/z_score/{name}-original-random-800.json'
        watermark_z_score_files = f'nlp-watermarking-main/result/ContextLS/M4/z_score/{name}-watermark-random-800.json'
        z_score_file = f'nlp-watermarking-main/result/ContextLS/M4/z_score/{name}-random-800.json'

        print(f"""
                ==============================
                  file_name => {file_name}
                  name => {name}
                  original_file => {original_file}
                  watermark_file => {watermark_file}
                  z_score_file => {z_score_file}
                ==============================
            """)

        # print(f'======{name}水印嵌入======')
        # original_output, watermark_output = embed(file_name, original_file, watermark_file, name)
        

        original_output = json.load(open(original_file))
        watermark_output = json.load(open(watermark_file))
        ori_fast_z_score = json.load(open(original_z_score_files))

        # print(f'======{name}原始文本提取======')
        # ori_fast_z_score = extract_watermark(original_output, original_z_score_files)
        print(f'======{name}水印文本提取======')
        water_fast_z_score = extract_watermark(watermark_output, watermark_z_score_files)

        z_score_output = []

        for index in range(len(ori_fast_z_score)):
            z_score_output.append({
                'text-index': ori_fast_z_score[index]['text-index'],
                'sentence-length': ori_fast_z_score[index]['sentence-length'],
                'ori-fast-z-score': ori_fast_z_score[index]['z-score'],
                'water-fast-z-score': water_fast_z_score[index]['z-score']
            })

        os.makedirs(os.path.dirname(z_score_file), exist_ok=True)
        with open(z_score_file, 'w', encoding='utf-8') as f:
            json.dump(z_score_output, f, indent=4, ensure_ascii=False)
        
        print(f'======\n{name}全部完成\n======')