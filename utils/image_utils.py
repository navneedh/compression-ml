import numpy as np

def transpose_image(img):
    return img.permute(1,2,0)