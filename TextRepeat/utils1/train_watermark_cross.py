import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
from torch.optim import lr_scheduler
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = out + x 
        return out

class TransformModel(nn.Module):
    def __init__(self, input_dim=768, num_layers=4, hidden_dim=500, output_dim=300):
        super(TransformModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(ResidualBlock(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        return x

# 计算句子嵌入之间的余弦相似度
def cosine_similarity(x, y):
    '''
    :param x: 句子嵌入
    :param y: 句子嵌入
    '''
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = torch.norm(x, p=2, dim=-1)
    norm_y = torch.norm(y, p=2, dim=-1)
    return dot_product / (norm_x * norm_y)

# 计算模型输出中行列均值的和
def row_col_mean_penalty(output):
    '''
    :param output:  模型变换后的输出
    '''
    row_mean_penalty = torch.mean(output, dim=1).pow(2).sum()
    col_mean_penalty = torch.mean(output, dim=0).pow(2).sum()
    return row_mean_penalty + col_mean_penalty

def abs_value_penalty(output):
    penalties = torch.relu(0.05 - torch.abs(output))
    mask = (penalties > 0).float()
    non_zero_count = torch.max(mask.sum(), torch.tensor(1.0))
    return (penalties * mask).sum() / non_zero_count

def vector_transform(vec):
    mean = vec.mean(dim=0, keepdim=True)
    centered_x = vec - mean
    transformed_x = torch.tanh(30 * centered_x)
    return transformed_x

def loss_fn(output_a, output_b, input_a, input_b, labels, lambda1=0.1, lambda2=1, median_value=0.4):
    '''
    :param output_a: 嵌入a经模型变换后的输出
    :param output_b: 嵌入b经模型变换后的输出
    :param input_a: 嵌入a
    :param input_b: 嵌入b
    '''
    original_similarity = cosine_similarity(input_a, input_b)
    original_similarity = torch.tanh(20 * (original_similarity - median_value))
    transformed_similarity = cosine_similarity(output_a, output_b)
    original_loss = torch.abs(original_similarity - transformed_similarity).mean()  # 语义一致性损失，即变化前后嵌入间余弦相似度的差值

    mean_penalty = row_col_mean_penalty(output_a) + row_col_mean_penalty(output_b)
    
    range_penalty_a = abs_value_penalty(output_a)
    range_penalty_b = abs_value_penalty(output_b)

    # 引入交叉熵损失
    cross_entropy_loss = F.cross_entropy(output_a, labels)

    total_loss = original_loss + lambda1 * mean_penalty + lambda2 * (range_penalty_a + range_penalty_b) + cross_entropy_loss
    return total_loss

def cosine_similarity_matrix(batch):
    norm = torch.norm(batch, dim=1).view(-1, 1)
    normed_batch = batch / norm
    similarity = torch.mm(normed_batch, normed_batch.t())
    return similarity

# 计算余弦相似度的中值
def get_median_value_of_similarity(all_token_embedding):
    similarity = cosine_similarity_matrix(all_token_embedding)
    median_value = torch.median(similarity)
    print(median_value)
    return median_value

class VectorDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]

if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_directory, '..', 'train_data', 'train_embeddings-bert_large.txt')
    output_path = os.path.join(current_directory, '..', 'models', 'transform_model-bert_large.pth')
    input_dim = 1024

    parser = argparse.ArgumentParser(description="Detect watermark in texts")
    parser.add_argument("--input_path", type=str, default=input_path)
    parser.add_argument("--output_model", type=str, default=output_path)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--input_dim", type=int, default=input_dim)

    args = parser.parse_args()
    embedding_data = np.loadtxt(args.input_path)
    print(embedding_data.size)
    data = torch.tensor(embedding_data, device='cuda', dtype=torch.float32)
    print(data.shape)
    median_value = get_median_value_of_similarity(data)
    # 假设我们有一些标签数据
    labels = torch.randint(0, 2, (len(data),))  # 随机生成一些标签作为示例
    dataset = VectorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformModel(input_dim=args.input_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    epochs = args.epochs
    for epoch in range(epochs):
        batch_iterator = iter(dataloader)
        
        for _ in range(len(dataloader) // 2):  
            input_a, labels_batch = next(batch_iterator)
            input_a = input_a.to(device)
            labels_batch = labels_batch.to(device)
            input_b = input_a  # 这里假设input_a和input_b相同，实际应用中可能不同
            
            output_a = model(input_a)
            output_b = model(input_b)
            loss = loss_fn(output_a, output_b, input_a, input_b, labels_batch, median_value=median_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if _ % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{_ + 1}/{len(dataloader) // 2}], Loss: {loss.item()}")

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(model.state_dict(), args.output_model)
    print('===模型训练完成===')