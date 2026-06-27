import numpy as np
import tensorflow as tf

from model import build_model
from train import train_model
from evaluate import evaluate_model, metrics_table
from optimizers import OPTIMIZER_NAMES


#CONFIG
SEEDS = [42, 7, 123]   # repeats per optimizer so we can average out the noise


def _set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_baseline(data: dict, epochs: int = 150, batch_size: int = 32,
                   seed: int = 42) -> dict:
    """
    The existing model: SGD, no class weights, no imbalance handling.

    This is the thing we are trying to beat. It scores well on raw
    accuracy but misses a large share of churners, which is exactly the
    failure mode that hurts a retention campaign.
    """
    _set_seed(seed)
    model = build_model(data['X_train'].shape[1], optimizer_name='sgd')
    info = train_model(
        model, data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        class_weights=None, epochs=epochs, batch_size=batch_size,
    )
    metrics = evaluate_model(model, data['X_test'], data['y_test'], name='baseline (sgd, no weights)')
    metrics['epochs_run'] = info['epochs_run']
    metrics['train_seconds'] = info['train_seconds']
    return {'metrics': metrics, 'history': info['history'], 'model': model}


def run_experiment(data: dict, optimizer_names: list = OPTIMIZER_NAMES,
                   seeds: list = SEEDS, epochs: int = 150, batch_size: int = 32) -> dict:
    """
    Train each optimizer once per seed with class weights, then aggregate.

    The first seed's models and histories are kept for the plots and for
    saving the winner. The aggregated table reports mean and std so a
    close race is not mistaken for a real gap.
    """
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    class_weights = data['class_weights']
    input_dim = X_train.shape[1]

    per_seed = {name: [] for name in optimizer_names}
    first_histories, first_models, first_results = {}, {}, []

    for name in optimizer_names:
        print(f"\n  Optimizer: {name}")
        for seed in seeds:
            _set_seed(seed)
            model = build_model(input_dim, optimizer_name=name)
            info = train_model(
                model, X_train, y_train, X_val, y_val,
                class_weights=class_weights, epochs=epochs, batch_size=batch_size,
            )
            m = evaluate_model(model, X_test, y_test, name=name)
            m['epochs_run'] = info['epochs_run']
            m['train_seconds'] = info['train_seconds']
            per_seed[name].append(m)

            if seed == seeds[0]:
                first_histories[name] = info['history']
                first_models[name] = model
                first_results.append(m)

    # Aggregate mean and std across seeds for each optimizer.
    agg_rows = []
    for name in optimizer_names:
        runs = per_seed[name]
        row = {'Optimizer': name}
        for key, label in [('accuracy', 'Accuracy'), ('precision', 'Precision'),
                           ('recall', 'Recall'), ('f1', 'F1'),
                           ('roc_auc', 'ROC AUC'), ('loss', 'Loss')]:
            vals = [r[key] for r in runs]
            row[label] = round(float(np.mean(vals)), 4)
            row[f'{label} std'] = round(float(np.std(vals)), 4)
        row['Epochs'] = int(np.mean([r['epochs_run'] for r in runs]))
        row['Train s'] = round(float(np.mean([r['train_seconds'] for r in runs])), 1)
        agg_rows.append(row)

    import pandas as pd
    agg_table = pd.DataFrame(agg_rows)

    # Winner by mean ROC AUC. AUC is threshold free, so it ranks how well
    # the model separates churners from the rest regardless of where the
    # campaign sets its cutoff.
    best_idx = int(np.argmax([r['ROC AUC'] for r in agg_rows]))
    best_name = agg_rows[best_idx]['Optimizer']

    return {
        'agg_table': agg_table,
        'per_seed': per_seed,
        'first_histories': first_histories,
        'first_models': first_models,
        'first_results': first_results,
        'best_name': best_name,
        'best_model': first_models[best_name],
        'y_test': y_test,
    }
