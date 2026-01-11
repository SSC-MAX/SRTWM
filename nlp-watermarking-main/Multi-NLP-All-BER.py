import json
import os
import random
from utils.misc import compute_ber
import numpy as np
import nltk
from tqdm import tqdm

nltk.data.path.append('/root/autodl-tmp/nltk')

def merge_extracted_messages(items):
    # 取最大长度
    max_len = max(len(item) for item in items)

    result = []
    for col in range(max_len):
        ones = zeros = 0

        for item in items:
            msg = item
            if col < len(msg):
                if msg[col] == 1:
                    ones += 1
                else:
                    zeros += 1

        # 取多数（若相等，这里默认取 1）

        result.append(1 if ones > zeros else 0)

    return result

def get_matrix_ber(dataset, target_bit, compare_type):
    matrix_ber_list = []
    for i in range(0, len(dataset)):
        sent = dataset[i]
        matrix_message = merge_extracted_messages(sent[f'{compare_type}-fast-extracted-message'])
        matrix_ber = get_ber(target_bit, matrix_message)
        dataset[i]['matrix-fast-ber'] = matrix_ber
        dataset[i]['matrix-fast-extracted-message'] = matrix_message
        matrix_ber_list.append(matrix_ber)
    return dataset, matrix_ber_list

def get_ber(target_bit, message_bit):
    error_cnt = 0
    check_len = min(len(target_bit), len(message_bit))
    max_len = max(len(target_bit), len(message_bit))
    for i in range(check_len):
        if target_bit[i] != message_bit[i]:
            error_cnt += 1
    error_cnt += max_len - check_len
    return error_cnt / len(target_bit) if error_cnt<len(target_bit) else 1

def get_agg_ber(dataset, compare_type=''):
    agg_ber_list = []
    for index in range(len(dataset)):
        bit_error_agg = dataset[index][f'{compare_type}-bit-error-agg']
        dataset[index]['agg-ber'] = bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt'] if bit_error_agg['sentence_cnt'] > 0 else 1
        agg_ber_list.append(dataset[index]['agg-ber'])
    return dataset, agg_ber_list

if __name__ == '__main__':
    file_name = '../result/RTW-NLP/HC3-random-800-NLP-Edit-1010-100-len.json'
    result_file = '../result/RTW-NLP/z_score/HC3-random-800-NLP-Edit-1010-100-ALL-BER.txt'
    len_file = '../result/RTW-NLP/HC3-random-800-NLP-Edit-1010-100-len.json'
    cnt_file = '../result/RTW-NLP/HC3-random-800-NLP-Edit-1010-100-cnt.json'
    f1_len_file = '../result/RTW-NLP/z_score/HC3-random-800-NLP-1010-100-ALL-BER-len-cnt.txt'

    target_bit = [1, 0, 1, 0]
    compare_type = 'water'
    dataset = json.load(open(file_name, 'r'))
    dataset_len = json.load(open(len_file, 'r'))
    dataset_cnt = json.load(open(cnt_file, 'r'))

    message_cnt = 0
    error_cnt_total = 0

    ber_list = []
    for i in tqdm(range(len(dataset))):
        # for item in dataset[i][0]:
        for item in dataset[i]:
            ber_list.append(item['water-fast-ber'])


    len_list = []
    for i in tqdm(range(len(dataset_len))):
        for item in dataset_len[i]:
            len_list.append(item['water-fast-ber'])
    for i in tqdm(range(len(dataset_cnt))):
        message_cnt += dataset_cnt[i]['message_cnt']
        error_cnt_total += dataset_cnt[i]['error_cnt_total']


    avg_ber = np.array(ber_list, dtype=float).mean()
    avg_ber_len = np.array(len_list, dtype=float).mean()

    f1_data = f'======\nDataset\nAvg_Ber:{avg_ber}\n======'

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(f1_data, f, indent=4, ensure_ascii=False)

    f1_len_data =  f'Avg_ber: {avg_ber_len}\nagg_ber:{error_cnt_total/message_cnt:.3f}'
    with open(f1_len_file, 'w', encoding='utf-8') as f:
        f.write(f1_len_data)

    print(f"======\nAvg_ber: {avg_ber_len}\nagg_ber:{error_cnt_total/message_cnt:.3f}======")