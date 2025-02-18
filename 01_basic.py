import torch

# x = torch.rand(2, 2)
# y = torch.rand(2, 2)
# print(torch.add(x, y))
# print(x + y)
# y.add_(x)
# print(y)

# x = torch.rand(5, 3)
# print(x)
# print(x[1, 1])
# print(x[1, 1].item())


x = torch.rand(4, 4)
print(x)
print(x.view(16))
print(x.view(2, 8))
z = x.view(-1, 8)
print(z)
print(z.size())
