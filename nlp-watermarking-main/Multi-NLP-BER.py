import json
import numpy as np
import os

def get_agg_ber(extracted_message, embed_message):
    total_ber = 0
    total_message = 0

    for i in range(len(embed_message)):

        total_message += len(embed_message[i])

    for i in range(len(extracted_message)):
        extracted_item = extracted_message[i]
        embed_item = embed_message[i]
        min_len = min(len(extracted_item), len(embed_item))
        max_len = max(len(extracted_item), len(embed_item))
        for j in range(min_len):
            if extracted_item[j] != embed_item[j]:
                total_ber += 1
        total_ber += max_len - min_len

    return total_ber/total_message if total_message > 0 else 1

def merge_extracted_messages(items):
    # 取最大长度
    max_len = max(len(item) for item in items[0])

    result = []
    for col in range(max_len):
        ones = zeros = 0

        for item in items[0]:
            msg = item
            if col < len(msg):
                if msg[col] == 1:
                    ones += 1
                else:
                    zeros += 1

        # 取多数（若相等，这里默认取 1）
        result.append(1 if ones >= zeros else 0)

    return result

def get_matrix_ber(extracted_message, embed_message):
    matrix_extracted_message = merge_extracted_messages(extracted_message)
    matrix_embed_message = merge_extracted_messages(embed_message)
    min_len = min(len(matrix_extracted_message), len(matrix_embed_message))
    max_len = max(len(matrix_extracted_message), len(matrix_embed_message))

    total_ber = 0
    total_message = len(matrix_embed_message)
    for i in range(min_len):
        if matrix_extracted_message[i] != matrix_embed_message[i]:
            total_ber += 1
    total_ber += max_len - min_len
    return total_ber/total_message if total_message > 0 else 1

def get_matrix_ber_1(extracted_message, target_bits):
    matrix_extracted_message = merge_extracted_messages(extracted_message)
    total_ber = 0
    total_message = min(len(matrix_extracted_message), len(target_bits))
    for i in range(total_message):
        if matrix_extracted_message[i] != target_bits[i]:
            total_ber += 1
    total_ber += max(len(matrix_extracted_message), len(target_bits))- min(len(matrix_extracted_message), len(target_bits))
    return total_ber / total_message if total_message > 0 else 1

def get_avg_ber(ber_list):
    ber_arr = np.array(ber_list, dtype='float')
    return ber_arr.mean()



if __name__ == '__main__':
    file_name = "../result/MULTI-NLP/delete/HC3-random-800-NLP-Edit-1010-100-1-z.json"
    result_file = "../result/MULTI-NLP/delete/HC3-random-800-NLP-Edit-1010-100-Edit-1-z.json"
    f1_file = "../result/MULTI-NLP/delete/HC3-random-800-NLP-Edit-1010-100-Edit-1-z.txt"
    extracted_compare_type = "delete"
    target_bits = [1,0,1,0]


    dataset = json.load(open(file_name))
    result = []
    matrix_ber_list = []
    agg_ber_list = []

    for index in range(len(dataset)):
        embed_message = []
        extracted_message = []
        item = dataset[index]
        for sent_item in item:
            embed_message.append(sent_item["water-embed-message"])
            extracted_message.append(sent_item[f"{extracted_compare_type}-fast-extracted-message"])

        agg_ber = get_agg_ber(extracted_message, embed_message)
        agg_ber_list.append(agg_ber)

        # result.append({
        #     "text-index": item["text-index"],
        #     "sentence-length": item["sentence-length"],
        #     "water-fast-embed-message": item["water-fast-embed-message"],
        #     "water-fast-extracted-message": item[f"{extracted_compare_type}-fast-extracted-message"],
        #     "water-fast-agg-ber": agg_ber
        # })

    # avg_matrix_ber = get_avg_ber(matrix_ber_list)
    avg_agg_ber = get_avg_ber(agg_ber_list)
    result_text =  f'agg_ber:{avg_agg_ber:.3f}'
    print(f'======\n{result_text}\n======')
    with open(f1_file, 'w') as f:
        f.write(result_text)




