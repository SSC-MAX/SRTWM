import json
from tqdm import tqdm

file_name = 'nlp-watermarking-main/result/nlp/ber/HC3-random-800.json'
dataset = json.load(open(file_name))

ber = 0
for index in tqdm(range(len(dataset))):
    text = dataset[index]
    text_ber = 0
    for sentence in text:
        text_ber += sentence['ber']
    ber += text_ber / len(text)
print(f'======\nber = {ber} / {len(dataset)} => {ber / len(dataset)}\n======')
