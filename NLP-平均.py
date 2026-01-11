import json
import numpy as np
import os

if __name__ == '__main__':
    file_name = 'result/nlp/HC3/HC3-random-800-100-1010-ALL-BER.json'
    f1_file = 'result/nlp/HC3/HC3-random-800-100-1010-ALL-BER.txt'

    dataset = json.load(open(file_name, 'r'))
    matrix_fast_ber_list = []
    agg_ber_list = []
    for i in range(0, len(dataset)):
        matrix_fast_ber_list.append(dataset[i]['matrix-fast-ber'])
        agg_ber_list.append(dataset[i]['agg-ber'])

    matrix_ber = np.array(matrix_fast_ber_list, dtype=float)  # 注意一定要是 float！
    matrix_avg_ber = matrix_ber.mean()

    agg_ber = np.array(agg_ber_list, dtype=float)
    agg_avg_ber = agg_ber.mean()

    print(f""" ================\nmatrix_avg_ber:{matrix_avg_ber:.3f}\nagg_avg_ber:{agg_avg_ber}================""")

    f1_data = f'matrix_ber:{matrix_avg_ber:.3f}\nagg_avg_ber:{agg_avg_ber:.3f}'
    with open(f1_file, 'w', encoding='utf-8') as file:
        file.write(f1_data)
