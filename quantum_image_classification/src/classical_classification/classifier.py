import os
import pickle
import numpy as np
from .traditional_models import train_svm, train_random_forest, evaluate_model

class ClassifierPipeline:
    def __init__(self):
        """Initialize classifier pipeline"""
        self.models = {}
        self.results = {}
    
    def train_models(self, X_train, y_train):
        """Train SVM and Random Forest models"""
        print("Training SVM model...")
        self.models['SVM'] = train_svm(X_train, y_train)
        
        print("Training Random Forest model...")
        self.models['RandomForest'] = train_random_forest(X_train, y_train)
        
        return self.models
    
    def evaluate_models(self, X_test, y_test, reports_dir='./reports'):
        """Evaluate trained models"""
        for name, model in self.models.items():
            self.results[name] = evaluate_model(model, X_test, y_test, name, reports_dir)
        
        return self.results
    
    def save_models(self, models_dir='./models'):
        """Save trained models"""
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(models_dir, f"{name.lower()}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} model to {model_path}")
    
    def load_models(self, models_dir='./models'):
        """Load trained models"""
        loaded_models = {}
        
        for name in ['SVM', 'RandomForest']:
            model_path = os.path.join(models_dir, f"{name.lower()}_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    loaded_models[name] = pickle.load(f)
                print(f"Loaded {name} model from {model_path}")
        
        if loaded_models:
            self.models = loaded_models
        
        return self.models
