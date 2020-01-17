import constants as c
import struct
import os
from PIL import Image
import numpy as np


def read(path=c.SAMPLE_PATH):
    x = []
    y = []
    size = []
    with open(path, mode='rb') as f:
        header_content = f.read(c.HEADER_SIZE)
        while header_content:
            _, tag_1, tag_2, width, height = struct.unpack(c.HEADER_FORMAT, header_content)
            tag = tag_1+tag_2
            # print(tag.decode('gbk'), _, width, height)
            image_size = width*height
            image_content = f.read(image_size)
            image_data = struct.unpack(f'<{image_size}B', image_content)
            # print(image_data)
            header_content = f.read(c.HEADER_SIZE)
            x.append(image_data)
            y.append(tag)
            size.append((height, width))
    return x, y, size


def integrate(path=c.TRAIN_PATH):
    ret_x = []
    ret_y = []
    file_list = os.listdir(path)
    for file_name in file_list:
        file_path = os.path.join(path, file_name)
        x, y, size = read(file_path)
        for i, j in zip(x,y):
            ret_x.extend(i)
            ret_y.extend(j)
    return ret_x, ret_y


def train_data():
    return integrate()


def test_data():
    return integrate(path=c.TEST_PATH)


def show_img_test():
    x, y, size = read()
    print('It is supposed to show', y[0].decode('gbk'))
    x_0 = np.array(x[0], dtype=np.uint8).reshape(size[0]) # PIL can't handle int64 array
    im = Image.fromarray(x_0)
    im.show()
