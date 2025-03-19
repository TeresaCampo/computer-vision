import random
import numpy as np
import torch
from skimage import data

im = data.astronaut()
im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
im = torch.from_numpy(im)
nbin = random.randint(32,128)

im = im.to(torch.float32)
im_bins = im * nbin // 256
hist_c1 = torch.histc(im_bins[0], bins = nbin, min = 0, max = nbin-1)
hist_c2 = torch.histc(im_bins[1], bins = nbin, min = 0, max = nbin-1)
hist_c3 = torch.histc(im_bins[2], bins = nbin, min = 0, max = nbin-1)

out = torch.cat((hist_c1,hist_c2,hist_c3)) / im.numel()