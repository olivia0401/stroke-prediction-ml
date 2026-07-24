"""Model performance visualization (SHAP-free)"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

sns.set_style('whitegrid')


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_proba, save_path='results/roc_curve.png'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] ROC curve saved to {save_path}")


def plot_feature_importance(model, feature_names, save_path='results/feature_importance.png'):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance', fontweight='bold', fontsize=14)
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Feature importance saved to {save_path}")


def plot_metrics_summary(metrics, save_path='results/metrics_summary.png'):
    """Plot metrics summary"""
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metric_names, metric_values,
                   color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('Model Performance Metrics', fontweight='bold', fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Metrics summary saved to {save_path}")


def create_all_visualizations(model, X_test, y_test, y_pred, y_proba, metrics):
    """Create all visualization plots"""
    print("\n" + "=" * 60)
    print("Creating Visualizations...")
    print("=" * 60 + "\n")

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba[:, 1])
    plot_feature_importance(model, X_test.columns.tolist())
    plot_metrics_summary(metrics)

    print("\n" + "=" * 60)
    print("[OK] All visualizations saved to results/ directory")
    print("=" * 60 + "\n")
