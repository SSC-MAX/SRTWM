import json
from tqdm import tqdm

file_name = 'result/nlp/HC3/watermark/HC3-random-200.json'
dataset = json.load(open(file_name))

sentences = []
for index in tqdm(range(len(dataset))):
    sentence = []
    for sen in dataset[index]:
        sentence.append({
            "c_idx": sen['c_idx'],
            "sen_idx": sen['sen_idx'],
            "sub_idset": sen['sub_idset'],
            "sub_idx": sen['sub_idx'],
            "clean_wm_text": "An auto quarter panel is a panel that was located on the quarter of a vehicle, which is the rear portion of the vehicle that extends from the back of the doors to the tail lights.",
            "key": "was",
            "msg": [
                1
            ]
        })
    sentences.append(sentence)

