import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
)



#CONFIG
CLASS_NAMES = ['Retain', 'Churn']   # 0 = stays, 1 = at risk segment
THRESHOLD = 0.5                     # same cut for every optimizer, fair comparison


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, name: str = '') -> dict:
    """
    Score one trained model on the held out test set.

    Returns the headline metrics plus the raw probabilities so the
    confusion matrix and ROC curve can reuse them.
    """
    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= THRESHOLD).astype(int)

    metrics = {
        'name': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'loss': float(tf.keras.losses.binary_crossentropy(
            y_test.astype(float), y_proba).numpy().mean()),
        'y_proba': y_proba,
        'y_pred': y_pred,
    }
    return metrics


def metrics_table(results: list) -> pd.DataFrame:
    """Stack the per optimizer metrics into one comparison table."""
    rows = []
    for r in results:
        rows.append({
            'Optimizer': r['name'],
            'Accuracy': round(r['accuracy'], 4),
            'Precision': round(r['precision'], 4),
            'Recall': round(r['recall'], 4),
            'F1': round(r['f1'], 4),
            'ROC AUC': round(r['roc_auc'], 4),
            'Loss': round(r['loss'], 4),
        })
    return pd.DataFrame(rows)


def print_report(model, X_test, y_test, name: str) -> None:
    """Full per class breakdown for the chosen best model."""
    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= THRESHOLD).astype(int)
    print(f"\n  Classification report for {name}:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))


def plot_optimizer_comparison(table: pd.DataFrame,
                              save_path: str = 'outputs/optimizer_comparison.png') -> None:
    """Grouped bar chart of the four headline metrics per optimizer."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    x = np.arange(len(table))
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 2) * width, table[m], width, label=m, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(table['Optimizer'], fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.set_title('Test Metrics by Optimizer', fontsize=13, pad=12)
    ax.legend(ncol=5, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Optimizer comparison saved to '{save_path}'.")


def plot_loss_curves(histories: dict, save_path: str = 'outputs/loss_curves.png') -> None:
    """Validation loss per epoch for every optimizer on one axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
    for (name, h), color in zip(histories.items(), palette):
        ax.plot(h.history['val_loss'], label=name, linewidth=2, color=color)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.set_title('Validation Loss by Optimizer', fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Loss curves saved to '{save_path}'.")


def plot_confusion(y_test, y_pred, name: str,
                   save_path: str = 'outputs/confusion_matrix.png') -> None:
    """Confusion matrix heatmap for the winning model."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='white', ax=ax)
    ax.set_title(f'Confusion Matrix ({name})\nTest Set', fontsize=12, pad=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Confusion matrix saved to '{save_path}'.")


def plot_roc(results: list, y_test,
             save_path: str = 'outputs/roc_curves.png') -> None:
    """ROC curve per optimizer so the AUC gap is visible, not just tabulated."""
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
    for r, color in zip(results, palette):
        fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
        ax.plot(fpr, tpr, label=f"{r['name']} (AUC={r['roc_auc']:.3f})",
                linewidth=2, color=color)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves by Optimizer (Test Set)', fontsize=13, pad=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] ROC curves saved to '{save_path}'.")
