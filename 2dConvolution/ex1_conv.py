import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, H, W, dtype=torch.float32)
kernel = torch.rand(oC, iC, kH, kW, dtype=torch.float32)

# exercise begin
n = input.shape[0]
iC = input.shape[1]
H = input.shape[2]
W = input.shape[3]

oC = kernel.shape[0]
kH = kernel.shape[2]
kW = kernel.shape[3]

# reshape input and kernel to allow boradcasting
input = torch.reshape(input, (n, 1, iC, H, W))
kernel = torch.reshape(kernel, (1, oC, iC, kH, kW))

# create output tensor
oH = H - (kH - 1)
oW = W - (kW - 1)
out = torch.zeros(n, oC, oH, oW)
out_i = 0

for i in range(H * W):
    i_col = i % W
    i_row = i // W

    # if kernel can't slide no more along x or y, continue
    if (i_row + kH > H) or (i_col + kW > W):
        continue

    # calculate output index
    out_col = out_i % oW
    out_row = out_i // oW
    out_i += 1

    # convolute kernel and input section, set result in output
    out[:, :, out_row, out_col] = (input[:, :, :, i_row : i_row + kH, i_col:i_col + kW] * kernel).sum(dim=-1).sum(dim=-1).sum(dim=-1)

