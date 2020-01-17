import os
import numpy as np
import utils.constants as c
from PIL import Image


def load_data(*args):
    X = []
    y = []
    for path in args:
        if os.path.isdir(path):
            class_tags = os.listdir(path)
            for class_tag in class_tags:
                class_data_path = os.path.join(path, class_tag)
                data_files = os.listdir(class_data_path)
                for data_file in data_files:
                    data_file_path = os.path.join(class_data_path, data_file)
                    img = Image.open(data_file_path)
                    data = np.array(img.getdata()).reshape(-1)
                    X.append(data)
                    y.append(class_tag)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_data(c.TRAIN_FILES_PATH)
    print(X.shape, y.shape)



