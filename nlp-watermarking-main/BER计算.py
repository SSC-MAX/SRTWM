import json
from tqdm import tqdm

file_name = 'nlp-watermarking-main/result/nlp/ber/HC3-random-800.json'
dataset = json.load(open(file_name))

total_ber = 0
sentence_length = 0
for text in dataset:
    text_ber = 0
    for sentence in text:
        ber = sentence['ber'] if sentence['error'] == 1 else 1
        total_ber += ber
    sentence_length += len(sentence)

print(f'average_ber = {total_ber} / {sentence_length} => {total_ber / sentence_length}')
