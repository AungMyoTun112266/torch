import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 2
# z = z.mean()
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)  # dz/dx
print(x.grad)


# # You can also stop autograd from tracking history on Tensors
x = torch.randn(3, requires_grad=True)
# Step 1
# x.requires_grad_(False)
print(x)

# Step 2
# y = x.detach()
# print(y)

# Step 3
with torch.no_grad():
    y = x + 2
    print(y)


# Step 4
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()
