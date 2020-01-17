import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR,'data','HWDB1.1trn_gnt')
TEST_PATH = os.path.join(BASE_DIR,'data','HWDB1.1tst_gnt')
SAMPLE_PATH = os.path.join(BASE_DIR,'sample','1001-f.gnt')
SAMPLE_PATH2 = os.path.join(BASE_DIR,'sample','001-f.gnt')
SAMPLE_PATH3 = os.path.join(BASE_DIR,'sample','501-f.gnt')

TEST_FILES_PATH = os.path.join(BASE_DIR,'data','test')
VALIDATION_FILES_PATH = os.path.join(BASE_DIR,'data','validation')
TRAIN_FILES_PATH = os.path.join(BASE_DIR,'data','train')

HEADER_FORMAT = '<I2c2H'
HEADER_SIZE = 10

RESIZE_WIDTH = 32
RESIZE_HEIGHT = 32

RESIZE = (RESIZE_WIDTH, RESIZE_HEIGHT)
