import os, sys, json, time
import numpy as np
import tensorflow as tf

from model import build_model
from train import train_model
from evaluate import evaluate_model

SEEDS = [42, 7]
EPOCHS = 60
BATCH = 64
OUT = 'outputs'
RESULTS = os.path.join(OUT, 'results.json')


def load_data():
    d = np.load(os.path.join(OUT, 'data_cache.npz'))
    meta = json.load(open(os.path.join(OUT, 'data_meta.json')))
    cw = {int(k): float(v) for k, v in meta['class_weights'].items()}
    return d, cw


def load_results():
    if os.path.exists(RESULTS):
        return json.load(open(RESULTS))
    return []


def save_result(rec):
    res = load_results()
    res.append(rec)
    json.dump(res, open(RESULTS, 'w'), indent=2)


def done_pairs():
    return {(r['name'], r['seed']) for r in load_results()}


def run_one(name, seed, d, cw):
    np.random.seed(seed); tf.random.set_seed(seed)
    is_baseline = (name == 'baseline')
    opt = 'sgd' if is_baseline else name
    weights = None if is_baseline else cw

    model = build_model(d['X_train'].shape[1], optimizer_name=opt)
    info = train_model(model, d['X_train'], d['y_train'], d['X_val'], d['y_val'],
                       class_weights=weights, epochs=EPOCHS, batch_size=BATCH)
    m = evaluate_model(model, d['X_test'], d['y_test'], name=name)

    rec = {
        'name': name, 'seed': seed,
        'accuracy': m['accuracy'], 'precision': m['precision'],
        'recall': m['recall'], 'f1': m['f1'], 'roc_auc': m['roc_auc'],
        'loss': m['loss'], 'epochs_run': info['epochs_run'],
        'train_seconds': round(info['train_seconds'], 1),
    }

    # For the first seed, keep extra artifacts for plotting and saving.
    if seed == SEEDS[0]:
        np.save(os.path.join(OUT, f'proba_{name}.npy'), m['y_proba'])
        json.dump(info['history'].history.get('val_loss', []),
                  open(os.path.join(OUT, f'valloss_{name}.json'), 'w'))
        if not is_baseline:
            model.save(os.path.join(OUT, f'model_{name}.keras'))
    save_result(rec)
    print(f"  saved {name} seed={seed}: f1={m['f1']:.4f} auc={m['roc_auc']:.4f} "
          f"recall={m['recall']:.4f} ({info['epochs_run']} ep)")


def main():
    names = sys.argv[1:]
    if not names:
        print("give optimizer names"); return
    d, cw = load_data()
    already = done_pairs()
    for name in names:
        for seed in SEEDS:
            if (name, seed) in already:
                print(f"  skip {name} seed={seed} (done)")
                continue
            run_one(name, seed, d, cw)


if __name__ == '__main__':
    main()
