import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # stop the float reordering that shifts results
os.environ["PYTHONHASHSEED"] = "0"

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from preprocessing import run_preprocessing
from experiment import train_baseline, run_experiment
from evaluate import (
    print_report, plot_optimizer_comparison, plot_loss_curves,
    plot_confusion, plot_roc,
)


#CONFIG
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data', 'teleconnect.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.keras')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')
SEED = 42

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: preprocessing
    data = run_preprocessing(DATASET_PATH, scaler_save_path=SCALER_PATH)

    # Step 2: baseline existing model
    print("\n" + "=" * 55)
    print("  BASELINE (existing model: sgd, no class weights)")
    print("=" * 55)
    base = train_baseline(data, epochs=150, seed=SEED)
    bm = base['metrics']
    print(f"  acc={bm['accuracy']:.4f} prec={bm['precision']:.4f} "
          f"recall={bm['recall']:.4f} f1={bm['f1']:.4f} auc={bm['roc_auc']:.4f}")

    # Step 3: optimizer sweep with class weights
    print("\n" + "=" * 55)
    print("  OPTIMIZER SWEEP (class weighted, 3 seeds each)")
    print("=" * 55)
    exp = run_experiment(data, epochs=150, batch_size=32)

    # Step 4: results table
    print("\n" + "=" * 55)
    print("  AGGREGATED TEST RESULTS (mean over seeds)")
    print("=" * 55)
    show = exp['agg_table'][['Optimizer', 'Accuracy', 'Precision', 'Recall',
                             'F1', 'ROC AUC', 'Loss', 'Epochs', 'Train s']]
    print(show.to_string(index=False))
    print(f"\n  Best optimizer by mean ROC AUC: {exp['best_name']}")

    # Save the full table for the report.
    exp['agg_table'].to_csv(os.path.join(OUTPUT_DIR, 'results_table.csv'), index=False)

    # Step 5: plots (built from the first seed models)
    print("\n" + "-" * 55)
    plot_df = exp['agg_table'][['Optimizer', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']]
    plot_optimizer_comparison(plot_df, os.path.join(OUTPUT_DIR, 'optimizer_comparison.png'))
    plot_loss_curves(exp['first_histories'], os.path.join(OUTPUT_DIR, 'loss_curves.png'))
    plot_roc(exp['first_results'], exp['y_test'], os.path.join(OUTPUT_DIR, 'roc_curves.png'))

    best = exp['best_model']
    best_pred = next(r for r in exp['first_results'] if r['name'] == exp['best_name'])['y_pred']
    plot_confusion(exp['y_test'], best_pred, exp['best_name'], os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    print_report(best, data['X_test'], exp['y_test'], exp['best_name'])

    # Step 6: save winner
    best.save(MODEL_PATH)
    print(f"\n  Saved best model ({exp['best_name']}) to '{MODEL_PATH}'.")

    # Step 7: baseline vs best summary
    best_row = exp['agg_table'][exp['agg_table']['Optimizer'] == exp['best_name']].iloc[0]
    print("\n" + "=" * 55)
    print("  EXISTING MODEL  vs  OPTIMIZED MODEL")
    print("=" * 55)
    print(f"  Accuracy : {bm['accuracy']:.4f}  ->  {best_row['Accuracy']:.4f}")
    print(f"  Recall   : {bm['recall']:.4f}  ->  {best_row['Recall']:.4f}")
    print(f"  F1       : {bm['f1']:.4f}  ->  {best_row['F1']:.4f}")
    print(f"  ROC AUC  : {bm['roc_auc']:.4f}  ->  {best_row['ROC AUC']:.4f}")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    set_global_seed(SEED)
    main()
