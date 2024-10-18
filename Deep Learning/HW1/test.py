import torch
data = torch.tensor(torch.arange(0, 5).view(1,1,1,5), requires_grad=True,dtype=torch.float32)
weight = torch.tensor(torch.arange(0,3).view(1,1,1,3),requires_grad=True,dtype=torch.float32)
#bias = pt.tensor(pt.zeros(1),requires_grad=True,dtype=torch.float32)
result = torch.nn.functional.conv2d(data,weight)
print(data)
print(weight)

result.backward(result.data)
print(data.grad)
print(weight.grad)