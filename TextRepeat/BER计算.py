import json
from tqdm import tqdm

file_name = 'result/REPEAT-MULTI/HC3/ber/HC3-random-800-multi-random.json'
dataset = json.load(open(file_name))

ber = 0
for index in tqdm(range(len(dataset))):
    text = dataset[index]
    text_ber = 0
    for sentence in text:
        text_ber += sentence['error_bit']
    ber += text_ber
print(f'======\nber = {ber} / {len(dataset)} => {ber / len(dataset)}\n======')
