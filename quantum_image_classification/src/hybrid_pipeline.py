# src/hybrid_pipeline.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

class HybridPipeline:
    def __init__(self, n_components=4, use_pca=True):
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = None  # Don't instantiate here
        self.scaler = StandardScaler()
        
    def run(self, X_train, y_train, X_test, y_test):
        # Preprocess training data first
        X_train = self._preprocess(X_train, fit_pca=True)
        
        # Preprocess test data with fitted PCA
        X_test = self._preprocess(X_test, fit_pca=False)
        
        # Rest of the pipeline remains the same
        X_train_q = self.qcnn.extract_features(X_train)
        X_test_q = self.qcnn.extract_features(X_test)
        
        model = Pipeline([
            ('scaler', self.scaler),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale'))
        ])
        
        model.fit(X_train_q, y_train)
        return model.score(X_test_q, y_test)

    def _preprocess(self, images, fit_pca=False):
        """Handle PCA fitting/transforming properly"""
        if self.use_pca:
            if fit_pca or self.pca is None:
                # Initialize and fit PCA on training data
                self.pca = PCA(self.n_components)
                return self.pca.fit_transform(images)
            else:
                # Transform using already fitted PCA
                try:
                    return self.pca.transform(images)
                except NotFittedError:
                    raise RuntimeError("PCA used before fitting. Call _preprocess with fit_pca=True first")
        return images
