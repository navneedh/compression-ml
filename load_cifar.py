import matplotlib.pyplot as plt

from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

x = unpickle("cifar/data_batch_5")[b'data']


for i in range(10000):
	first_image = x[i].reshape(3, 32,32)

	Image.fromarray(first_image.transpose(1,2, 0)).save("cifar_data/{}.png".format(i + 40000))


