from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def train_svm(X_train, y_train):
    """Train SVM model with hyperparameter tuning"""
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }
    
    svm = GridSearchCV(SVC(probability=True), param_grid, cv=3)
    svm.fit(X_train, y_train)
    
    print(f"Best SVM parameters: {svm.best_params_}")
    return svm.best_estimator_

def train_random_forest(X_train, y_train):
    """Train Random Forest model with hyperparameter tuning"""
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }
    
    rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    rf.fit(X_train, y_train)
    
    print(f"Best Random Forest parameters: {rf.best_params_}")
    return rf.best_estimator_

def evaluate_model(model, X_test, y_test, model_name, reports_dir='./reports'):
    """Evaluate model and save results"""
    # Create reports directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print metrics
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(os.path.join(reports_dir, f'{model_name.lower()}_confusion_matrix.png'))
    
    # Save metrics to file
    with open(os.path.join(reports_dir, f'{model_name.lower()}_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
