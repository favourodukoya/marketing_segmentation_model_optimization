import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from optimizers import get_optimizer


#CONFIG
DEFAULT_OPTIMIZER = 'adam'
L2_LAMBDA = 1e-4   # mild weight penalty, keeps the dense weights from drifting large


def build_model(input_dim: int, optimizer_name: str = DEFAULT_OPTIMIZER,
                verbose: bool = False) -> tf.keras.Model:
    """
    Build and compile the network for one optimizer.

    The 64 -> 32 -> 16 funnel gives the first layer room to mix the 30
    input columns before the network compresses toward the single churn
    probability. Three things hold overfitting down: dropout, batch
    normalisation on the wider layers, and a small L2 penalty on the dense
    weights. We track accuracy, AUC, precision and recall during training,
    not just accuracy, because on an imbalanced target accuracy alone hides
    how the model treats the churn class.
    """
    model = Sequential(name='churn_segmentation_ann', layers=[
        Input(shape=(input_dim,), name='input'),

        # Hidden layer 1
        Dense(64, activation='relu', kernel_regularizer=l2(L2_LAMBDA), name='dense_1'),
        BatchNormalization(name='batchnorm_1'),
        Dropout(0.3, name='dropout_1'),

        # Hidden layer 2
        Dense(32, activation='relu', kernel_regularizer=l2(L2_LAMBDA), name='dense_2'),
        BatchNormalization(name='batchnorm_2'),
        Dropout(0.2, name='dropout_2'),

        # Hidden layer 3
        Dense(16, activation='relu', kernel_regularizer=l2(L2_LAMBDA), name='dense_3'),
        Dropout(0.1, name='dropout_3'),

        # Output: one sigmoid neuron for churn probability.
        Dense(1, activation='sigmoid', name='output'),
    ])

    model.compile(
        optimizer=get_optimizer(optimizer_name),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )

    if verbose:
        model.summary()
    return model


def get_callbacks() -> list:
    """
    Callbacks shared by every run.

    No ModelCheckpoint here because the experiment trains several models and
    keeps each one in memory, then saves only the winner from main.py.
    EarlyStopping restores the best weights so a run that starts to overfit
    still returns its best version.
    """
    early_stop = EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7,
        min_lr=1e-6, verbose=0
    )
    return [early_stop, reduce_lr]