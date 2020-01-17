import numpy as np
from models import knn, svm, cnn
from sklearn.metrics import accuracy_score
import utils.load_data as load_data
import utils.constants as c

def knn_driver():
    ks = [3, 4, 5, 6, 7]
    X, y = load_data.load_data(c.TRAIN_FILES_PATH, c.VALIDATION_FILES_PATH)
    for k in ks:
        accuracy, k, train_cost = knn.knn_train(X, y, k)
        print(f'TRAIN: k: {k}; Acurracy: {np.mean(accuracy)}; time cost: {train_cost*1000}ms')

    k = 5
    X_test, y_test = load_data.load_data(c.TEST_FILES_PATH)
    pred_y, test_cost = knn.knn_test(X, y, k, X_test)
    accuracy = accuracy_score(y_test, pred_y)
    print(f'TEST: k: {k}; Acurracy: {accuracy}; time cost: {test_cost*1000}ms')


def svm_driver():
    kernel = 'rbf'
    X, y = load_data.load_data(c.TRAIN_FILES_PATH, c.VALIDATION_FILES_PATH)
    accuracy, params, train_cost = svm.svm_train(X, y, kernel=kernel)
    print(f'TRAIN: params: {params}; Kernel: {kernel}; Acurracy: {accuracy}; time cost: {train_cost*1000}ms')

    params = {
        'C': 1e5,
        'gamma': 1e-8,
    }
    X_test, y_test = load_data.load_data(c.TEST_FILES_PATH)
    pred_y, test_cost = svm.svm_test(X, y, X_test, C=params['C'], gamma=params['gamma'], kernel=kernel)
    accuracy = accuracy_score(y_test, pred_y)
    print(f'TEST: params: {params}; Acurracy: {accuracy}; time cost: {test_cost*1000}ms')



def cnn_driver():
    print('Please run cnn.py in the models directory directly')
    


if __name__ == '__main__':
    knn_driver()
    svm_driver()
    cnn_driver()