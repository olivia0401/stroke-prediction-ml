"""SHAP model interpretability"""
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Run: pip install shap")


def plot_shap_summary(model, X_test, save_path='results/shap_summary.png'):
    """Generate SHAP feature importance plot
    
    Args:
        model: Trained model
        X_test: Test data
        save_path: Save path
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available, skipping visualization")
        return
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"SHAP summary saved to {save_path}")
