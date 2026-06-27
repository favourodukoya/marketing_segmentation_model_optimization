# Otomoto Marketing Segmentation: ANN Optimizer Comparison

This project recreates a marketing segmentation model for Otomoto and then optimizes it. The task is framed as churn-risk segmentation: split the customer base into an at-risk group worth contacting and a stable group that can stay on routine handling. The data is the TeleConnect customer file, the IBM Telco Customer Churn sample, with 7,043 customers and a churn rate of about 26.5 percent.

The core of the work is a comparison of five optimization algorithms on a fixed neural network. Everything else about the model is held constant so the optimizer is the only thing that changes between runs. RMSProp comes out ahead and is saved as the deployed model.

## What the project does

The brief does not describe Otomoto's existing model, so the pipeline first builds a reasonable baseline to stand in for it: a standard feed-forward network trained with plain stochastic gradient descent and no class weighting. That baseline scores well on accuracy but misses almost half the churners, which is the failure mode a retention campaign cannot afford.

The optimization happens in two moves. First, balanced class weights so the rarer churn class pulls more weight during training. Second, a sweep over five optimizers (SGD, SGD with Nesterov momentum, RMSProp, Adam, Nadam), each run three times with different seeds and averaged. The winner is chosen by mean ROC AUC, since AUC measures how well the model ranks churners above stayers regardless of where the decision threshold is later set.

## Results in short

The saved RMSProp model lifts churn recall from 0.56 to 0.80 against the baseline, raises F1 from 0.59 to 0.62, and nudges AUC from 0.841 to 0.844. In plain terms, it catches 223 of 280 churners on the test set where the baseline caught 157. Accuracy and precision drop a little, which is the expected trade for catching far more churners. The five optimizers finish within about a point of each other on every test metric, so the honest reading is that the class weighting did most of the work, and the clearest separation between optimizers is convergence speed: RMSProp settles in 17 epochs where plain SGD needs about 150.

## Project structure

```
module6_ANN/
├── data/
│   └── teleconnect.csv          # IBM Telco Customer Churn sample (place here)
├── outputs/                     # created on first run (models, scaler, figures, tables)
├── tests/
│   └── test_pipeline.py         # 31 pytest tests covering the whole pipeline
├── preprocessing.py             # load, clean, encode, split, scale, class weights
├── model.py                     # the 64-32-16 network and the shared callbacks
├── optimizers.py                # the five optimizers, keyed by name
├── train.py                     # one training run with timing
├── evaluate.py                  # metrics and all the plotting functions
├── experiment.py                # baseline plus the three-seed optimizer sweep
├── main.py                      # runs everything end to end
├── extra_figs.py                # the four figures main.py does not produce
├── run_chunk.py                 # optional chunked runner for slow machines
└── pyproject.toml               # dependencies and Python version
```

### What each file does

`preprocessing.py` does all the data work. It fills the 11 blank `TotalCharges` values (all of them brand-new customers with tenure 0) with zero, drops the `customerID` column, maps the binary columns to 0/1, one-hot encodes the ten multi-level categoricals with the first level dropped, and ends with a 30-column feature matrix. The split is a stratified 70/15/15 train, validation, and test, and only the three continuous columns (`tenure`, `MonthlyCharges`, `TotalCharges`) are standardized. The scaler is fit on the training split alone and saved to `outputs/scaler.pkl`, so no test statistics leak backward. Balanced class weights are computed from the training labels.

`model.py` builds the network: three hidden layers of 64, 32, and 16 ReLU units, batch normalization on the first two layers, dropout of 0.3, 0.2, and 0.1, and a single sigmoid output for churn probability. Every dense layer carries a small L2 penalty of 1e-4. The model is compiled to track accuracy, AUC, precision, and recall together, since accuracy alone hides how the churn class is treated. The callbacks are early stopping with a patience of 15 that restores the best weights, and a learning-rate reducer that halves the rate after seven flat epochs.

`optimizers.py` holds the five optimizers, all at a base learning rate of 0.001, returned fresh by name.

`experiment.py` runs the baseline and then the sweep. Each optimizer trains once per seed (42, 7, 123), and the results are averaged so a single lucky start cannot pass for a real edge. The winner is the optimizer with the highest mean ROC AUC.

`evaluate.py` scores a model on the test set and draws the figures: the optimizer comparison bars, the validation loss curves, the ROC curves, and the confusion matrix.

`main.py` is the entry point. It sets the seeds, runs preprocessing, trains the baseline, runs the sweep, prints the aggregated table, saves the figures and the winning model, and prints the before-and-after summary.

`extra_figs.py` produces the four figures `main.py` does not: the before/after chart, the train-versus-validation curves, the precision-recall and threshold sweep, and the permutation feature importance. For the before/after, threshold, and importance plots it loads the saved `best_model.keras` so those figures match the confusion matrix exactly.

## Setup

This project uses Python 3.12. With `uv`:

```bash
uv sync
```

Or with pip:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib pytest
```

Place `teleconnect.csv` in the `data/` folder before running.

## How to run

Run the full pipeline:

```bash
uv run main.py
```

This writes the trained model, the scaler, the results table, and the four main figures into `outputs/`.

Generate the four extra figures (run this after `main.py`, since it reads the saved model):

```bash
uv run extra_figs.py
```

Run the tests:

```bash
uv run pytest tests/test_pipeline.py -v
```

All 31 tests should pass.

## Tests

The suite is grouped by stage. It checks that the file loads with the right row count and both classes, that `TotalCharges` becomes numeric and the blanks are filled rather than dropped, that the encoded target is binary and the churn rate stays near 26.5 percent, that the class balance holds across the splits, that the scaled numeric columns come out near zero mean with no NaNs, that each optimizer builds and an unknown name raises an error, that the network outputs a single probability in range, that the dense layers carry the L2 penalty, that training tracks AUC, precision, and recall, and that the evaluation metrics sit in the unit range and match the threshold.

## Reproducibility

The run is seeded and reproducible. `main.py` sets the Python, NumPy, and TensorFlow seeds and turns off the oneDNN floating-point reordering (`TF_ENABLE_ONEDNN_OPTS=0`) that otherwise shifts results slightly between runs. Small differences can still appear across different machines, which is why the report treats the optimizer ranking as a close race rather than a decisive win.

## Notes

