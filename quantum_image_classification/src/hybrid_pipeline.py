import numpy as np
from data.datasets import MNISTDataHandler
from src.preprocessing.image_preprocessing import normalize_images
from src.quantum_processing.brqi_encoding import create_brqi_circuit
from src.quantum_processing.feature_extraction import extract_features
from src.classical_classification.traditional_models import *

class HybridPipeline:
    def __init__(self):
        self.data_handler = MNISTDataHandler()
        
    def run(self, sample_size=100):
        # Load data
        X_train, y_train, X_test, y_test = self.data_handler.get_numpy_data()
        
        # Process subset
        X_train = normalize_images(X_train[:sample_size])
        X_test = normalize_images(X_test[:sample_size])
        
        # Quantum feature extraction
        train_features = [self.process_image(img) for img in X_train]
        test_features = [self.process_image(img) for img in X_test]
        
        # Classical training
        svm = train_svm(train_features, y_train[:sample_size])
        rf = train_random_forest(train_features, y_train[:sample_size])
        
        # Evaluation
        return {
            'SVM': svm.score(test_features, y_test[:sample_size]),
            'RandomForest': rf.score(test_features, y_test[:sample_size])
        }
    
    def process_image(self, img):
        circuit = create_brqi_circuit(img.reshape(28, 28))
        return extract_features(circuit)[:32]  # Use first 32 features
