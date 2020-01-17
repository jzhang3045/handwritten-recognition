# Handwriting Recognition for Chinese Characters

Term project for STAT MACHINE LEARNING at GaTech

Other team members: [@halimiqi](https://github.com/halimiqi), [@BarryXIONG1996](https://github.com/BarryXIONG1996), [@Junpeng Zhang](https://github.com/jzhang3045) , [@Yulong Gu]()

## Dataset
We use the isolated character datasets HWDB1.1 (offline) provided by National Laboratory of Pattern Recognition (NLPR)
Institute of Automation of Chinese Academy of Sciences
Credits to http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html

Each file contains ~3000 characters which are written by one person. The data of one character includes `data_size`, 
`tag_1`, `tag_2`, `image_width`, `image_height` and `image_binary_content`.
 
For example, the first character in sample `/sample/1001-f.gnt` is `!`. The data we got is as below.
```python
data_size = 946 # the whole data size of current char including the header info
tag_1 = b'!' # first byte
tag_2 = b'\x00' # second byte
tag = tag_1 + tag_2 # the type of a Chinese character is char which has 2 bytes
image_width = 12
image_height = 78
```

The image of the handing writing of this character is like

![sample img](https://github.com/ian7yang/handwriting-recognition/blob/master/statics/sample.png?raw=true)

Then the image is resized to 32*32 pixel wise in order to have a small dimension.

## Code Implementation

The entrance of this application is `main.py`. There are three drivers that handle three models including `K-Nearest Neighbor`,
`Support Vector Machine` and `Convontional Nerual Network (using ResNet framework)`.

In this code repo, we just upload some sample images, including training, validation and testing data.
