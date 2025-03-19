from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# open image with PIL
im = Image.open("test_image.jpg")
transform = transforms.ToTensor()
im_tensor = transform(im)                       # C, H, W

# show image with matplot lib
im_np = im_tensor.permute(1, 2, 0).numpy()      # H, C, W
plt.imshow(im_tensor)
plt.axis("off")
plt.show()