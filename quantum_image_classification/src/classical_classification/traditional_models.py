from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }
    svm = GridSearchCV(SVC(), param_grid, cv=3)
    svm.fit(X_train, y_train)
    return svm.best_estimator_

def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10]
    }
    rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
    rf.fit(X_train, y_train)
    return rf.best_estimator_
