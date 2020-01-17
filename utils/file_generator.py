"""
    This script is used to extract binary data and classify them into corresponding folders
    folder name is the class name
    files inside the folder is the characters written by different people
"""
import os
import numpy as np
from PIL import Image

import constants as c
from data_extractor import read


def generate_file(data_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_list = os.listdir(data_path)
    for file_name in file_list:
        file_path = os.path.join(data_path, file_name)
        x, y, size = read(file_path)
        for i, jks in enumerate(zip(x, y, size)):
            j, k, s = jks
            folder_name = k.decode('gbk')
            folder_path = os.path.join(output_path, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path, f'{i}.jpg')
            a = np.array(j,dtype=np.uint8).reshape(s)
            img = Image.fromarray(a).resize(c.RESIZE)
            img = img.convert(mode='L')
            img.save(file_path)
