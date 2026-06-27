import time
import numpy as np
import tensorflow as tf
from model import get_callbacks


def train_model(model: tf.keras.Model,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                class_weights: dict = None,
                epochs: int = 150, batch_size: int = 32) -> dict:
    """
    class_weights are passed straight to Keras so the rarer churn class
    pulls more weight during training. We also time the run so the report
    can compare how quickly each optimizer settles.
    """
    callbacks = get_callbacks()

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks, verbose=0,
    )
    elapsed = time.time() - start

    epochs_run = len(history.history['loss'])
    best_val_loss = float(np.min(history.history['val_loss']))
    best_val_auc = float(np.max(history.history['val_auc']))

    print(f"    ran {epochs_run} epochs in {elapsed:.1f}s | "
          f"best val_loss={best_val_loss:.4f} | best val_auc={best_val_auc:.4f}")

    return {
        'history': history,
        'epochs_run': epochs_run,
        'train_seconds': elapsed,
        'best_val_loss': best_val_loss,
        'best_val_auc': best_val_auc,
    }
