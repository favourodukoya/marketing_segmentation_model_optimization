from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Nadam


#CONFIG
BASE_LR = 0.001


def get_optimizer(name: str):
    """Return a new optimizer instance for the given name."""
    name = name.lower()

    if name == 'sgd':
        # Plain mini batch gradient descent, no momentum. The baseline.
        return SGD(learning_rate=BASE_LR)

    if name == 'momentum':
        # SGD with Nesterov momentum to push through shallow regions.
        return SGD(learning_rate=BASE_LR, momentum=0.9, nesterov=True)

    if name == 'rmsprop':
        # Per parameter adaptive rate, good for the sparse one hot columns.
        return RMSprop(learning_rate=BASE_LR)

    if name == 'adam':
        # Momentum plus adaptive rates, the usual strong default.
        return Adam(learning_rate=BASE_LR)

    if name == 'nadam':
        # Adam with Nesterov momentum, sometimes a touch faster to settle.
        return Nadam(learning_rate=BASE_LR)

    raise ValueError(f"Unknown optimizer '{name}'.")


# The exact set the experiment compares, baseline first.
OPTIMIZER_NAMES = ['sgd', 'momentum', 'rmsprop', 'adam', 'nadam']
