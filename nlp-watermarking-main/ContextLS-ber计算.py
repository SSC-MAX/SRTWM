import json
import os
import random
from utils.misc import compute_ber
import numpy as np
import nltk

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
        result.append(1 if ones >= zeros else 0)

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

def get_agg_ber(dataset):
    agg_ber_list = []
    for index in range(len(dataset)):
        bit_error_agg = dataset[index]['bit-error-agg']
        dataset[index]['agg-ber'] = bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt'] if bit_error_agg['sentence_cnt'] > 0 else 1
        agg_ber_list.append(dataset[index]['agg-ber'])
    return dataset, agg_ber_list

if __name__ == '__main__':
    file_name = '../result/ContextLS/HC3/delete/HC3-random-800-1010-100-e-1.json'
    result_file = '../result/ContextLS/HC3/delete/HC3-random-800-1010-100-e-1-ALL-BER.json'
    f1_file = '../result/ContextLS/HC3/delete/HC3-random-800-1010-100-e-1-ALL-BER.txt'

    target_bit = [1, 0, 1, 0]
    compare_type = 'delete'
    dataset = json.load(open(file_name, 'r'))
    dataset, matrix_ber_list = get_matrix_ber(dataset, target_bit, compare_type)
    dataset, agg_ber_list = get_agg_ber(dataset)
    avg_matrix_ber = np.array(matrix_ber_list, dtype=float).mean()
    avg_agg_ber = np.array(agg_ber_list, dtype=float).mean()

    avg_text = f'avg_matrix_ber:{avg_matrix_ber}\navg_agg_ber:{avg_agg_ber}'
    with open(f1_file, 'w', encoding='utf-8') as file:
        file.write(avg_text)

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"======\nMatrix Ber: {avg_matrix_ber}\nAgg Ber: {avg_agg_ber}\n======")