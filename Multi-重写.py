import json
import os
import sys
from openai import OpenAI
import time
from tqdm import tqdm

def get_total_num(dataset):
    count = 0
    for data in dataset:
        count += len(data)
    return count

def batch_format(dataset, batch_file):
    batch = []

    for index in range(len(dataset)):
        for j in range(len(dataset[index])):
            batch.append({
                "custom_id": f"request-{index}-{j}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user",
                         "content": f"""Rewrite the following paragraph:\n {dataset[index][j]["clean_wm_text"]}"""}
                    ],
                }
            })

    os.makedirs(os.path.dirname(batch_file), exist_ok=True)
    with open(batch_file, 'w', encoding='utf-8') as f:
        for item in batch:
            json.dump(item, f)
            f.write('\n')
    return get_total_num(dataset)

def rewrite_text(file_name, batch_file, result_file, client, name):
    print(f"""
                ==============================
                  file_name => {file_name}
                  result_file => {result_file}
                  name => {name}
                ==============================
                """)
    dataset = json.load(open(file_name))
    total_count = batch_format(dataset, batch_file)
    print('===batch文件创建完毕===')


    batch_input_file = client.files.create(
        file=open(batch_file, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    batch_created = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"context-{name}-rewrite"
        }
    )
    print(f'===batch创建完毕===\n{batch_created.id}')

def rewrite(file_name,result_file, client):
    dataset = json.load(open(file_name))[:100]

    output = []
    for i in tqdm(range(len(dataset))):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Your are a helpful assistant to rewrite the text."},
                {"role": "user", "content": f"""Rewrite the following paragraph slightly by replacing:\n {dataset[i]["watermark_text"]}"""},
            ],
        )
        rewrite_text =  response.choices[0].message.content
        output.append({
            "original_text": dataset[i]["original_text"],
            'watermark_text': dataset[i]["watermark_text"],
            "rewrite_text": rewrite_text,
            "ori-fast-z-score": dataset[i]["ori-fast-z-score"],
            "water-fast-z-score": dataset[i]["water-fast-z-score"],
            "ori-fast-bits-hat": dataset[i]["ori-fast-bits-hat"],
            "water-fast-bits-hat": dataset[i]["water-fast-bits-hat"],
            "ori-fast-ber": dataset[i]["ori-fast-ber"],
            "water-fast-ber": dataset[i]["water-fast-ber"]
        })

        time.sleep(0.05)

        with open(result_file, 'w', encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

    print('======改写完成======')


if __name__ == '__main__':
    client = OpenAI(
        api_key="sk-lwmtM6MJKF6rFYaM02A31e43721440E8B5808dC332B7D093",
        base_url="https://api.laozhang.ai/v1"
    )

    dataset_name = ['HC3']

    for name in dataset_name:
        try:
            # file_name = f'result/nlp/M4/watermark/dataset-M4-random-{name}-800.json'
            # batch_file_c = f'result/nlp/M4/translate/batch_file/nlp-{name}-random-800-c.jsonl'
            # result_file_c = f'result/nlp/M4/translate/nlp-{name}-random-800-c.json'
            # batch_file_e = f'result/nlp/M4/translate/batch_file/nlp-{name}-random-800-e.jsonl'
            # result_file_e = f'result/nlp/M4/translate/nlp-{name}-random-800-e.json'

            # file_name = f'result/Multibit/water/m4-random-{name}-800.json'
            # result_file = f'result/Multibit/rewrite/m4-random-{name}-800.json'
            file_name = f'result/MULTI-NONSEM/HC3-random-800-NonSemEdit-100.json'
            result_file = f'result/MULTI-NONSEM/rewrite/HC3-random-800-NonSemEdit-100.json'

            print(f'======\nfile_name => {file_name}\nresult_file => {result_file}\n======')


            rewrite(file_name, result_file, client)
            print('================')
        except BaseException as e:
            print(e)
