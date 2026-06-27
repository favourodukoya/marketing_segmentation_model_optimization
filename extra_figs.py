import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "0"

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, precision_recall_curve, average_precision_score,
                             precision_score, recall_score, f1_score)

from preprocessing import run_preprocessing
from model import build_model
from train import train_model
from evaluate import evaluate_model

OUT = 'outputs'
WINNER = 'rmsprop'        # change this if your run selects a different optimizer
SEED = 42


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s); tf.keras.utils.set_random_seed(s)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_seed()
    data = run_preprocessing(os.path.join('data', 'teleconnect.csv'),
                             scaler_save_path=os.path.join(OUT, 'scaler.pkl'))
    y_test = data['y_test']

    # Baseline: sgd, no class weights, for the before/after chart.
    np.random.seed(SEED); tf.random.set_seed(SEED)
    base = build_model(data['X_train'].shape[1], optimizer_name='sgd')
    train_model(base, data['X_train'], data['y_train'], data['X_val'], data['y_val'],
                class_weights=None, epochs=150, batch_size=32)
    bm = evaluate_model(base, data['X_test'], y_test, name='baseline')

    # Winner: retrain once to capture the training history for the curves.
    np.random.seed(SEED); tf.random.set_seed(SEED)
    win = build_model(data['X_train'].shape[1], optimizer_name=WINNER)
    info = train_model(win, data['X_train'], data['y_train'], data['X_val'], data['y_val'],
                       class_weights=data['class_weights'], epochs=150, batch_size=32)
    hist = info['history'].history

    # For the before/after, threshold, and importance figures, score the SAVED
    # model that main.py wrote out. That keeps these figures matched to the
    # confusion matrix, which also comes from the saved model.
    saved_path = os.path.join(OUT, 'best_model.keras')
    if os.path.exists(saved_path):
        win = tf.keras.models.load_model(saved_path)
    wm = evaluate_model(win, data['X_test'], y_test, name=WINNER)
    p = wm['y_proba']

    # FIG: before / after
    klab = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    bvals = [bm['accuracy'], bm['precision'], bm['recall'], bm['f1'], bm['roc_auc']]
    wvals = [wm['accuracy'], wm['precision'], wm['recall'], wm['f1'], wm['roc_auc']]
    x = np.arange(5); w = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, bvals, w, label='Existing model (SGD, no weights)', color='#B0B0B0')
    b2 = ax.bar(x + w/2, wvals, w, label=f'Optimized ({WINNER.upper()} + class weights)', color='#4C72B0')
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f'{b.get_height():.2f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(klab); ax.set_ylim([0, 1.0]); ax.set_ylabel('Score')
    ax.set_title('Existing Model vs Optimized Model (test set)'); ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/before_after.png', dpi=150, bbox_inches='tight'); plt.close()

    # FIG: train vs validation
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(hist['accuracy'], label='Train', linewidth=2, color='#4C72B0')
    ax[0].plot(hist['val_accuracy'], label='Validation', linewidth=2, linestyle='--', color='#DD8452')
    ax[0].set_title('Accuracy over Epochs'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(hist['loss'], label='Train', linewidth=2, color='#4C72B0')
    ax[1].plot(hist['val_loss'], label='Validation', linewidth=2, linestyle='--', color='#DD8452')
    ax[1].set_title('Loss over Epochs'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Binary Crossentropy'); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(f'{WINNER.upper()} Training History (seed {SEED})', y=1.02)
    plt.tight_layout(); plt.savefig(f'{OUT}/train_val_curves.png', dpi=150, bbox_inches='tight'); plt.close()

    # FIG: precision-recall and threshold sweep
    prec, rec, _ = precision_recall_curve(y_test, p); ap = average_precision_score(y_test, p); br = y_test.mean()
    ths = np.linspace(0.1, 0.9, 81); P = []; R = []; F = []
    for th in ths:
        q = (p >= th).astype(int)
        P.append(precision_score(y_test, q, zero_division=0)); R.append(recall_score(y_test, q, zero_division=0)); F.append(f1_score(y_test, q, zero_division=0))
    P, R, F = map(np.array, (P, R, F)); bt = ths[int(np.argmax(F))]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(rec, prec, color='#4C72B0', linewidth=2, label=f'{WINNER.upper()} (PR-AUC={ap:.3f})')
    ax[0].axhline(br, color='grey', linestyle='--', alpha=0.7, label=f'Base rate ({br:.2f})')
    ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision'); ax[0].set_title('Precision-Recall Curve (test set)'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ths, P, label='Precision', color='#55A868', linewidth=2); ax[1].plot(ths, R, label='Recall', color='#C44E52', linewidth=2); ax[1].plot(ths, F, label='F1', color='#8172B2', linewidth=2)
    ax[1].axvline(0.5, color='grey', linestyle=':', alpha=0.7, label='Default 0.5'); ax[1].axvline(bt, color='black', linestyle='--', alpha=0.7, label=f'Max F1 ({bt:.2f})')
    ax[1].set_xlabel('Decision Threshold'); ax[1].set_ylabel('Score'); ax[1].set_title('Metrics vs Threshold'); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/threshold_analysis.png', dpi=150, bbox_inches='tight'); plt.close()

    # FIG: permutation importance
    feat = data['feature_names']
    ba = roc_auc_score(y_test, win.predict(data['X_test'], verbose=0).ravel())
    rng = np.random.default_rng(SEED); imp = []
    for j in range(data['X_test'].shape[1]):
        Xp = data['X_test'].copy(); rng.shuffle(Xp[:, j])
        imp.append(ba - roc_auc_score(y_test, win.predict(Xp, verbose=0).ravel()))
    imp = np.array(imp); order = np.argsort(imp)[::-1][:10][::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([feat[j] for j in order], [imp[j] for j in order], color='#4C72B0')
    ax.set_xlabel('Drop in ROC AUC when feature is shuffled')
    ax.set_title(f'Top 10 Features by Permutation Importance ({WINNER.upper()})'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/feature_importance.png', dpi=150, bbox_inches='tight'); plt.close()

    print('\nSaved before_after.png, train_val_curves.png, threshold_analysis.png, feature_importance.png')
    print(f'PR-AUC={ap:.3f}, F1-max threshold={bt:.2f}')


if __name__ == '__main__':
    main()