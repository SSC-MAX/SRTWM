import numpy as np
import torch
import json
import os


def scale_vector(v):
    mean = np.mean(v)
    v_minus_mean = v - mean
    v_minus_mean = np.tanh(1000 * v_minus_mean)
    return v_minus_mean

# 输出文本嵌入
def get_embedding(device,embedding_model,embedding_tokenizer,sentence):
    input_ids = embedding_tokenizer.encode(sentence, return_tensors="pt", max_length=512, truncation="longest_first")
    input_ids = input_ids.to(device)
    with torch.no_grad():
        output = embedding_model(input_ids)
        # print(output)
    return output[0][:, 0, :]

def get_embedding_roberta(device, roberta_model, roberta_tokenizer, sentence):
    input_ids = roberta_tokenizer(sentence, return_tensors="pt", max_length=512, truncation="longest_first").to(device)
    # 使用模型获取句子的语义嵌入
    with torch.no_grad():
        outputs = roberta_model(**input_ids)
    # 句子的语义嵌入是最后一层隐藏状态的第一个token（CLS token）的输出
    # print(outputs)
    return outputs.last_hidden_state[:, 0, :]

def get_embedding_t5(device, t5_model, t5_tokenizer, sentence):
    sentence = 'summarize: '+sentence
    # 编码输入文本
    inputs = t5_tokenizer(sentence, return_tensors='pt', max_length=512, truncation='longest_first').to(device)
    decoder_input_ids = torch.full((inputs['input_ids'].shape[0], 1), t5_tokenizer.eos_token_id, dtype=torch.long, device=device)
    # 将编码后的输入传递给模型
    with torch.no_grad():
        outputs = t5_model(input_ids=inputs['input_ids'], decoder_input_ids=decoder_input_ids)
    # 提取最后一层的隐藏状态作为句子的语义嵌入
    return outputs.last_hidden_state[:, 0, :]

def get_embedding_bge(device, bge_model, bge_tokenizer, sentence):
    input_ids = bge_tokenizer.encode(sentence, return_tensors="pt", max_length=512, truncation="longest_first").to(device)
    # 使用模型获取句子的语义嵌入
    with torch.no_grad():
        outputs = bge_model(input_ids)
    # 句子的语义嵌入是最后一层隐藏状态的第一个token（CLS token）的输出
    # print(outputs)
    return outputs.last_hidden_state[:, 0, :]



def get_bias(device,embedding_tokenizer,embedding_model,transform_model,context_sentence,mapping):
    context_embedding = get_embedding(device,embedding_model,embedding_tokenizer,context_sentence)
    with torch.no_grad():
        output = transform_model(context_embedding).cpu()[0].detach().numpy()
    similarity_array = scale_vector(output)[mapping]
    # similarity_array = scale_vector(output)
    return torch.from_numpy(-similarity_array)

def cosine_similarity(x, y):
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = torch.norm(x, p=2, dim=-1)
    norm_y = torch.norm(y, p=2, dim=-1)
    return dot_product / (norm_x * norm_y)
