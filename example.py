import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

from functions import generate_blocks
from functions import extract_prnu_var


def var(imgs):
    print("generate blocks")
    blocks = generate_blocks(imgs[0].shape, randomize=False)

    print("extracting prnu")
    return extract_prnu_var(imgs, blocks, lum_range=(5, 250),  r=7, levels=1)


def rand_var_a(imgs):
    print("generate blocks")
    blocks = generate_blocks(imgs[0].shape, randomize=False)

    print("extracting prnu")
    return extract_prnu_var(imgs, blocks, lum_range=(5, 250), r=7, R=10, levels=1)


def rand_var_b(imgs):
    print("generate blocks")
    blocks = generate_blocks(imgs[0].shape, randomize=True)

    print("extracting prnu")
    return extract_prnu_var(imgs, blocks, lum_range=(5, 250), r=7, R=10, levels=1)


print("loading images")
dirlist = np.array(sorted(glob("data/*.jpg")))
imgs = []
for im_path in dirlist:
    im = Image.open(im_path)
    imgs.append(np.array(im, copy=False))

K = var(imgs)

print("show prnu")
print("\tdtype:", K.dtype)
print("\tmin max:", "%.2f" % K.min(), "%.2f" % K.max())
plt.imshow(K, cmap="Greys")
plt.show()
