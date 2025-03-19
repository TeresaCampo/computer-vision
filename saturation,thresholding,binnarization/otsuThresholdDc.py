import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = torch.from_numpy(im)

N = im.shape[0]*im.shape[1]
threshold=0
max_inter_class_variance=0

for i in range(1,256):
    w_1 = (im[im <= i].sum().sum()).item() /N
    w_2 = (im[im > i].sum().sum()).item() /N

    if(w_1==0 or w_2==0):
        continue
    m_1 = 0.0
    m_2 = 0.0
    for index in range(i+1):
        m_1 += im[im == index].sum().sum().item()*index
    for index in range(i+1,256):
        m_2 += im[im == index].sum().sum().item()*index

    m_1 /= w_1
    m_2 /= w_2

    inter_class_variance=w_1 * w_2 * pow(m_1-m_2,2)
    print('i='+str(i)+'variance='+str(inter_class_variance))

    if inter_class_variance > max_inter_class_variance:
        threshold = i
        max_inter_class_variance = inter_class_variance

print(threshold)




