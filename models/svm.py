import numpy as np
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV



def svm_train(X, y, kernel='rbf'):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    C_range = np.logspace(4, 6, 3)
    gamma_range = np.logspace(-6, -8, 3)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    param_grid = dict(gamma=gamma_range, C=C_range)
    start_at = time.time()
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    end_at = time.time()

    return grid.best_score_, grid.best_params_, end_at-start_at

def svm_test(train_X, train_y, test_X, C=1e5, gamma=1e-7, kernel='rbf'):
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    start_at = time.time()
    classifier = SVC(C=C, kernel=kernel, gamma=gamma, random_state=2019)
    classifier.fit(train_X, train_y)
    pred_y = classifier.predict(test_X)
    end_at = time.time()
    
    return pred_y, end_at-start_at