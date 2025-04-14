# main.py
import argparse
from src.hybrid_pipeline import HybridPipeline
from src.utils.experiment_tracker import ExperimentTracker
import json

def main():
    parser = argparse.ArgumentParser(description='Run Quantum Image Classification Pipeline')
    parser.add_argument('--feature-type', type=str, default='brqi',
                        choices=['brqi', 'quantum_circuit', 'classical', 'raw'],
                        help='Type of features to extract')
    parser.add_argument('--subset-size', type=int, default=1000,
                        help='Number of images to process (None for all)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--tune-hyperparams', action='store_true',
                        help='Whether to tune hyperparameters')
    parser.add_argument('--data-dir', type=str, default='./data/mnist',
                        help='Directory containing MNIST data')
    parser.add_argument('--processed-dir', type=str, default='./data/mnist/processed',
                        help='Directory to save processed data')
    parser.add_argument('--models-dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--reports-dir', type=str, default='./reports',
                        help='Directory to save reports')
    parser.add_argument('--experiments-dir', type=str, default='./experiments',
                        help='Directory to store experiment logs')
    parser.add_argument('--list-experiments', action='store_true',
                        help='List all previous experiments')
    parser.add_argument('--load-experiment', type=str,
                        help='Load and display results of a specific experiment')
    
    args = parser.parse_args()
    
    if args.list_experiments:
        experiment_tracker = ExperimentTracker(args.experiments_dir)
        experiments = experiment_tracker.list_experiments()
        print("Previous experiments:")
        for exp in experiments:
            print(f"  {exp}")
        return
    
    if args.load_experiment:
        experiment_tracker = ExperimentTracker(args.experiments_dir)
        experiment_data = experiment_tracker.load_experiment(args.load_experiment)
        print("Experiment configuration:")
        print(json.dumps(experiment_data['config'], indent=2))
        print("\nExperiment results:")
        print(json.dumps(experiment_data['results'], indent=2))
        return
    
    pipeline = HybridPipeline(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        experiments_dir=args.experiments_dir
    )
    
    results = pipeline.run_pipeline(
        feature_type=args.feature_type,
        subset_size=args.subset_size,
        test_size=args.test_size,
        tune_hyperparams=args.tune_hyperparams
    )
    
    print("Pipeline execution completed.")
    print("Results:")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for metric, value in model_results.items():
            if metric != 'confusion_matrix':
                print(f"  {metric}: {value:.4f}")
    
    print(f"\nDetailed reports and visualizations saved in {args.reports_dir}")
    print(f"Experiment logged in {args.experiments_dir}")

if __name__ == "__main__":
    main()