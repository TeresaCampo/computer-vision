import random
import numpy as np
import torch
from skimage import data
import matplotlib.pyplot as plt


im = data.astronaut()
im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
im = torch.from_numpy(im)
nbin = random.randint(32,128)

def binarize_color_channel(mono_color_channel_image : torch.tensor, channel : int, histogram : torch.tensor, nbin : int ):
    for row in mono_color_channel_image:
        for pixel in row:
            index = channel * nbin + (pixel.item() * nbin // 256)
            histogram[index]+=1

            # create normalized image

def binarize_color_channel_faster(mono_color_channel_image : torch.tensor, channel : int, histogram : torch.tensor, nbin : int ):
    im_bins = (mono_color_channel_image * nbin) // 256
    hist_channel = torch.histc(im_bins, bins= nbin,min = 0, max = nbin)
    histogram[channel * nbin : ((channel+1)*nbin)] =  hist_channel


def binarize_color_channel_create_image(mono_color_channel_image: torch.tensor, channel: int, nbin: int, new_image : torch.tensor):
    for i_row in range(mono_color_channel_image.shape[0]):
        for i_col in range(mono_color_channel_image.shape[1]):
            color = (mono_color_channel_image[i_row, i_col].item() * nbin // 256)
            new_image[channel, i_row, i_col] = color

def normalized_image_computation(im : torch.tensor, nbin : int):
    # normalized image computation
    out_image = torch.tensor(np.zeros((3, im.shape[1], im.shape[2]))).to(torch.float32)
    binarize_color_channel_create_image(im[0, ::], 0, nbin, out_image)
    binarize_color_channel_create_image(im[1, ::], 1, nbin, out_image)
    binarize_color_channel_create_image(im[2, ::], 2, nbin, out_image)

    # Convertiamo l'immagine elaborata in NumPy per la visualizzazione
    im_np = im.numpy().transpose(1, 2, 0) / 255.0  # Normalizziamo tra 0 e 1
    out_image_np = out_image.numpy().transpose(1, 2, 0)
    out_image_np = out_image_np / out_image_np.max()  # Normalizziamo per visualizzare bene i dati

    # Visualizzazione con Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(im_np)
    axes[0].set_title("Immagine Originale")
    axes[0].axis("off")

    axes[1].imshow(out_image_np)
    axes[1].set_title("Immagine Binarizzata")
    axes[1].axis("off")

    plt.show()

# normalized histogram computation
# by hand
out = torch.zeros( 3*nbin , dtype=torch.float32 )
im = im.to(torch.float32)

binarize_color_channel_faster(im[0,::], 0, out, nbin)
binarize_color_channel_faster(im[1,::], 1, out, nbin)
binarize_color_channel_faster(im[2,::], 2, out, nbin)
out = out / im.numel()

# extra to create a normalized image
# normalized_image_computation(im, nbin)







