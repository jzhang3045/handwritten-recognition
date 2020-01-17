import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
import time
from sklearn.model_selection import cross_val_score


def knn_train(X, y, k):
    start_at = time.time()
    classifier = KNN(n_neighbors=k, algorithm='brute',weights='distance')
    accuracy = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
    end_at = time.time()

    return accuracy, k, end_at-start_at

def knn_test(train_X, train_y, k, test_X):
    start_at = time.time()
    classifier = KNN(n_neighbors=k, algorithm='brute',weights='distance')
    classifier.fit(train_X, train_y)
    pred_y = classifier.predict(test_X)
    end_at = time.time()
    
    return pred_y, end_at-start_at

