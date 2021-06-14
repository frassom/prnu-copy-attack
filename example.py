import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

import prnu
from extract import extract_prnu_var

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

print("loading images")
dirlist = np.array(sorted(glob("data/*.jpg")))
imgs = []
for im_path in dirlist:
    im = Image.open(im_path)
    imgs.append(np.array(im, copy=False))

print("extracting prnu")
W = extract_prnu_var(imgs, levels=3, r=7, R=14,
                     t_low=10, t_up=245, rand_type="B")

print("show prnu")
print("\tdtype:", W.dtype)
print("\tmin max:", "%.2f" % W.min(), "%.2f" % W.max())
plt.imshow(W, cmap="Greys")
plt.show()
