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
    parser.add_argument('--data', type=str, default='data/stroke-data.csv')
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
    X, y, scaler, encoders = preprocess_data(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset: Train={len(X_train)}, Test={len(X_test)}\n")

    # Train
    trainer = ModelTrainer(model_type=args.model, use_mlflow=False)
    metrics = trainer.train(X_train, y_train, X_test, y_test)

    # Get predictions for visualization
    y_pred = trainer.model.predict(X_test)
    y_proba = trainer.model.predict_proba(X_test)

    # Save
    trainer.save_model(args.output)
    output_dir = Path(args.output).parent
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    joblib.dump(encoders, output_dir / 'encoders.pkl')

    # Results
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"F1 Score:   {metrics['f1_score']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"{'='*60}\n")

    # Visualizations
    if args.viz:
        create_all_visualizations(
            trainer.model, X_test, y_test, y_pred, y_proba, metrics
        )

    print(f"✓ Model saved to {args.output}\n")


if __name__ == '__main__':
    main()
