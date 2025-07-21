# main.py
import argparse
from data.datasets import MNISTDataHandler
from src.hybrid_pipeline import HybridPipeline
def main():
    parser = argparse.ArgumentParser(description='Quantum MNIST Classification')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--components', type=int, default=4)
    parser.add_argument('--use-pca', action='store_true')
    
    args = parser.parse_args()
    
    # Load data
    data_handler = MNISTDataHandler()
    X_train, y_train, X_test, y_test = data_handler.get_numpy_data()
    
    # Initialize and run pipeline
    pipeline = HybridPipeline(n_components=args.components, use_pca=args.use_pca)
    accuracy = pipeline.run(
        X_train[:args.samples], 
        y_train[:args.samples],
        X_test[:args.samples],
        y_test[:args.samples]
    )
    
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
