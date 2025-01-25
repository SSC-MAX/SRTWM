from transformers import AutoTokenizer, AutoModel
import torch

# 加载模型和分词器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(r"D:\model\sentence-transformers\all-MiniLM-L6-v2")
model = AutoModel.from_pretrained(r"D:\model\sentence-transformers\all-MiniLM-L6-v2").to(device)
# tokenizer = AutoTokenizer.from_pretrained(r"D:\model\sentence-transformers\stsb-roberta-base-v2")
# model = AutoModel.from_pretrained(r"D:\model\sentence-transformers\stsb-roberta-base-v2").to(device)
# 函数：计算文本嵌入
def get_embedding(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output[0][:,0].cpu().numpy()

# 计算两个文本的相似度
def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    cosine_similarity = torch.nn.functional.cosine_similarity(torch.tensor(embedding1), torch.tensor(embedding2))
    return cosine_similarity.item()


calculate_similarity("aaa","bbb")