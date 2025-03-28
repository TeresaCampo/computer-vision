import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(2, 3)
input = torch.rand((n, iC, H, W), dtype=torch.float32)

n = input.shape[0]
iC = input.shape[1]
H = input.shape[2]
W = input.shape[3]

outH = int((H - (kH - 1) - 1) / s) + 1
outW = int((W - (kW - 1) - 1) / s) + 1
out = torch.zeros((n, iC, outH, outW), dtype=torch.float32)

for i in range(0, H * W, s):
    i_row = i % W
    i_col = i // W
    if (i_row + kH) > H or (i_col + kW) > W or (i_row % s) != 0:
        continue

    out_row = int(i_row / s)
    out_col = int(i_col / s)
    out[:, :, out_row, out_col] = input[:, :, i_row:i_row + kH, i_col:i_col + kW].max(dim=-1)[0].max(dim=-1)[0]
