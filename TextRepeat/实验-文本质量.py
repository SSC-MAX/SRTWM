import json
import os
import torch
from tqdm import tqdm
from utils1 import bias
from watermark_REPEAT_NO_CONTEXT_BERT import watermark_model as repeat_no_context_bert_model

result_file = 'TextRepeat/trial-quality/result/HC3-random-800.json'
dataset = json.load(open('TextRepeat/trial-quality/HC3-random-800.json'))
device = torch.device('cuda')

repeat_model = repeat_no_context_bert_model()
tokenizer = repeat_model.embedding_tokenizer
model = repeat_model.embedding_model

count = {'0.9':0, '0.85':0,'0.8':0, '0.75':0, '0.7':0}
output = []

for index in tqdm(range(len(dataset))):
    data = dataset[index]
    original_text = data['original_text']
    watermark_text = data['watermark_text']
    original_embedding = bias.get_embedding(torch.device('cuda'), model, tokenizer, original_text)
    watermark_embedding = bias.get_embedding(torch.device('cuda'), model, tokenizer, watermark_text)
    similarity = bias.cosine_similarity(original_embedding, watermark_embedding)
    if similarity >= 0.9:
        count['0.9'] += 1
    if 0.85 <= similarity < 0.9:
        count['0.85'] += 1
    if 0.8 <= similarity < 0.85:
        count['0.8'] += 1
    if 0.75 <= similarity < 0.8:
        count['0.75'] += 1

output.append(count)
os.makedirs(os.path.dirname(result_file), exist_ok=True)
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(count)