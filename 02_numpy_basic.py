import torch
import numpy as np

# Converting a Torch Tensor to a NumPy Array
# The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.
print("Converting a Torch Tensor to a NumPy Array")
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))

a.add_(1)
print(a)
print(b)


# Converting NumPy Array to Torch Tensor
# See how changing the np array changed the Torch Tensor automatically
print("Converting NumPy Array to Torch Tensor")
z = np.ones(5)
print(z)
y = torch.from_numpy(z)
print(y)
z += 1
print(z)
print(y)


print("CUDA Tensors")
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
