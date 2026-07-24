"""CLI training script with visualizations"""
import argparse
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data
from src.preprocessor import preprocess_data
from src.trainer import ModelTrainer
from src.visualizer import create_all_visualizations
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description='Train stroke prediction model')
    parser.add_argument('--data', type=str, default=str(Path(__file__).resolve().parent.parent / 'data' / 'stroke-data.csv'))
    parser.add_argument('--model', choices=['rf', 'xgb'], default='rf')
    parser.add_argument('--output', type=str, default='models/model.pkl')
    parser.add_argument('--viz', action='store_true', help='Generate visualizations')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} Model")
    print(f"{'='*60}\n")

    # Load data
    df = load_data(args.data)

    # Preprocess
    X, y, preprocessor = preprocess_data(df)
    print(f"Dataset: {len(X)} samples\n")

    # Train: fit the final pipeline on all data and get an honest,
    # cross-validated performance estimate (see ModelTrainer.train).
    trainer = ModelTrainer(model_type=args.model, use_mlflow=False)
    metrics = trainer.train(X, y, preprocessor=preprocessor)

    # Get predictions for visualization (use full dataset)
    y_pred = trainer.predict(X)
    y_proba = trainer.predict_proba(X)

    # Save (skip saving preprocessor due to pickle issue with lambda functions)
    trainer.save_model(args.output)

    # Results (cross-validated, held-out estimate)
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE (5-fold cross-validated)")
    print(f"{'='*60}")
    print(f"F1 Score:   {metrics['f1_score']:.4f} +/- {metrics['f1_score_std']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f} +/- {metrics['precision_std']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f} +/- {metrics['recall_std']:.4f}")
    print(f"AUC-ROC:    {metrics['roc_auc']:.4f} +/- {metrics['roc_auc_std']:.4f}")
    print(f"Deployment threshold: {metrics['threshold']:.3f}")
    print(f"{'='*60}\n")

    # Visualizations
    if args.viz:
        create_all_visualizations(
            trainer.model, X, y, y_pred, y_proba, metrics
        )

    print(f"[OK] Model saved to {args.output}\n")


if __name__ == '__main__':
    main()
