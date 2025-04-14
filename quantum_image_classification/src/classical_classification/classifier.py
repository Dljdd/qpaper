# src/classical_classification/classifier.py
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ClassifierPipeline:
    def __init__(self):
        self.models = {
            'SVM': SVC(),
            'RandomForest': RandomForestClassifier()
        }
    
    def train_models(self, X_train, y_train):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            results[name] = accuracy_score(y_test, y_pred)
        return results
