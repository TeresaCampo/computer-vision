import numpy as np
from skimage import data
from skimage.transform import resize
import torch
import matplotlib.pyplot as plt


im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = torch.from_numpy(im).to(torch.float32)
N = im.numel()

fig, ax = plt.subplots(1,2)
# ---------------------handy histogram------------------------------------
hist = np.zeros(256)
for i in im:
    for j in i:
        hist[int(j.item())] += 1

# normalize histogram
hist_norm_handy = hist / N
print(f'Hist by hand--> {hist_norm_handy[:10]}')

ax[0].bar(range(256), hist_norm_handy)
ax[0].set_title("By hand")
ax[0].set_xlabel("Grey scale levels")
ax[0].set_ylabel("Frequency")

# ---------------------torch.histc() histogram---------------------------
hist = torch.histc(im, 256, max=255, min=0).numpy()
hist_norm_torch = hist / N
print(f'Hist with torch.histc()--> {hist_norm_torch[:10]}')
ax[1].bar(range(256), hist_norm_torch)
ax[1].set_title("torch.histc()")
ax[1].set_xlabel("Grey scale levels")
ax[1].set_ylabel("Frequency")

fig.show()

print(f'Difference in histograms--> {(hist_norm_torch-hist_norm_handy).sum()}')
