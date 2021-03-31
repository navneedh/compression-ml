from struct import unpack
import numpy as np

output_folder = "/media/expansion1/navneedhmaudgalya/Datasets/tiny_imagenet/test_jpeg_1/"

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}

total_len = 0

class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        global total_len
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                count = 0
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]
            if len(data)==0:
                break

if __name__ == "__main__":
    img = JPEG(output_folder + '1.png')
    img.decode()

    bpp = []

    for i in range(10000):
        with open(output_folder + "{}.png".format(i), "rb") as image:
            f = image.read()
            index = f.find(b'\xff\xda')


            bpp.append((8 * (len((bytearray(f))) - index))/(64 * 64))

    print(sum(bpp)/10000)
    np.save(output_folder + "bpp.npy", bpp)

# if __name__ == "__main__":
#     output_folder = "trans_data/test_jpeg2r_05/"
#     img = JPEG(output_folder + '1.jp2')
#     img.decode()
#
#     bpp = []
#
#     for i in range(10000):
#         with open(output_folder + "{}.jp2".format(i), "rb") as image:
#             f = image.read()
#             print(((bytearray(f))))
#
#             index = f.find(b'\x93')
#             bpp.append((8 * (len((bytearray(f))) - index))/(32 * 32))
#
#     print(sum(bpp)/10000)
#     np.save(output_folder + "bpp.npy", bpp)

# OUTPUT:
# Start of Image
# Application Default Header
# Quantization Table
# Quantization Table
# Start of Frame
# Huffman Table
# Huffman Table
# Huffman Table
# Huffman Table
# Start of Scan
# End of Image
