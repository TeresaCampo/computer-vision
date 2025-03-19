import numpy as np
from skimage import data
from skimage.transform import resize
import torch
import matplotlib.pyplot as plt

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = torch.from_numpy(im)

# find histogram
hist_norm = torch.histc(im.to(torch.float32), bins= 256, min = 0, max = 255) / im.numel()

plt.bar(range(256), hist_norm)
plt.title("Image histogram")
plt.xlabel("Grey scale levels")
plt.ylabel("Frequency")
plt.show()

# apply otsu thresholding
inter_class_variances = np.zeros(256)
for t in range(255):
    w_1 = hist_norm[:t+1].sum()
    w_2 = hist_norm[t+1:].sum()

    if w_1 * w_2 == 0:
        print(f'threshold {t} skipped')
        continue

    m_1 = (hist_norm[:t+1] * np.array(range(t+1))).sum()/w_1
    m_2 = (hist_norm[t+1:] * np.array(range(t+1, 256))).sum()/w_2

    inter_class_variances[t] = w_1 * w_2 * ( m_1 - m_2 )**2

out = inter_class_variances.argmax()
print(f'=======================')
print(f'Otsu threshold is {out}')
