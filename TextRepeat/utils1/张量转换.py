import torch

tensor1 = torch.tensor([[1,1],[1,2],[1,3],[1,4]])
print(f'tensor:{tensor1}\nshape:{tensor1.shape}')

output1 = tensor1[0:1]
output2 = tensor1[1:2]

for i in range(2, tensor1.shape[0], 2):
    output1 = torch.cat((output1, tensor1[i:i+1]), dim=0)
    output2 = torch.cat((output2, tensor1[i+1:i+2]), dim=0)

print(f'output1:{output1}\noutput2:{output2}')

# print(f'{torch.cat(output1, [1,2])}')