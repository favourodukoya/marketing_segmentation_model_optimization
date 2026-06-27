import os
import sys
import pytest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocessing import (
    load_data, clean_data, check_missing_values, encode_features,
    split_data, normalize_features, get_class_weights,
    TARGET_COL, ID_COL, NUMERIC_COLS,
)
from model import build_model
from optimizers import get_optimizer, OPTIMIZER_NAMES
from evaluate import evaluate_model, THRESHOLD

DATASET_PATH = os.path.join(ROOT, 'data', 'teleconnect.csv')


#FIXTURES
@pytest.fixture(scope='session')
def raw_df():
    return load_data(DATASET_PATH)


@pytest.fixture(scope='session')
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture(scope='session')
def encoded(clean_df):
    X, y = encode_features(clean_df)
    return X, y


@pytest.fixture(scope='session')
def splits(encoded):
    X, y = encoded
    return split_data(X, y)


@pytest.fixture(scope='session')
def scaled(splits, tmp_path_factory):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    path = str(tmp_path_factory.mktemp('art') / 'scaler.pkl')
    X_tr, X_va, X_te, scaler = normalize_features(
        X_train, X_val, X_test, scaler_save_path=path
    )
    return X_tr, X_va, X_te, y_train.values, y_val.values, y_test.values


@pytest.fixture(scope='session')
def small_model(scaled):
    """A 3 epoch model, only used to confirm evaluation works end to end."""
    X_tr, X_va, _, y_tr, y_va, _ = scaled
    model = build_model(input_dim=X_tr.shape[1], optimizer_name='adam')
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=3, batch_size=32, verbose=0)
    return model


#TESTS
class TestLoading:

    def test_file_exists(self):
        assert os.path.exists(DATASET_PATH)

    def test_loads_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)
        assert raw_df.shape[0] == 7043

    def test_target_present(self, raw_df):
        assert TARGET_COL in raw_df.columns

    def test_both_classes_present(self, raw_df):
        assert set(raw_df[TARGET_COL].unique()) == {'Yes', 'No'}


class TestCleaning:

    def test_totalcharges_is_numeric(self, clean_df):
        assert pd.api.types.is_numeric_dtype(clean_df['TotalCharges'])

    def test_no_missing_after_clean(self, clean_df):
        assert check_missing_values(clean_df).sum() == 0

    def test_id_dropped(self, clean_df):
        assert ID_COL not in clean_df.columns

    def test_blank_rows_filled_not_dropped(self, raw_df, clean_df):
        # Cleaning must not lose rows, the 11 blanks are filled in place.
        assert clean_df.shape[0] == raw_df.shape[0]


class TestEncoding:

    def test_target_is_binary(self, encoded):
        _, y = encoded
        assert set(np.unique(y)) == {0, 1}

    def test_all_features_numeric(self, encoded):
        X, _ = encoded
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        assert non_numeric == []

    def test_target_not_in_features(self, encoded):
        X, _ = encoded
        assert TARGET_COL not in X.columns

    def test_churn_rate_reasonable(self, encoded):
        # Sanity check on the known imbalance, about 26.5% positive.
        _, y = encoded
        assert 0.25 < y.mean() < 0.28


class TestSplitting:

    def test_total_conserved(self, splits):
        X_train, X_val, X_test, *_ = splits
        assert len(X_train) + len(X_val) + len(X_test) == 7043

    def test_stratification_holds(self, splits):
        # Positive rate should stay close across splits.
        _, _, _, y_train, y_val, y_test = splits
        rates = [y_train.mean(), y_val.mean(), y_test.mean()]
        assert max(rates) - min(rates) < 0.02

    def test_feature_count_matches(self, splits, encoded):
        X, _ = encoded
        X_train, *_ = splits
        assert X_train.shape[1] == X.shape[1]


class TestNormalization:

    def test_numeric_mean_near_zero(self, scaled, encoded):
        X, _ = encoded
        X_tr, *_ = scaled
        idx = [X.columns.get_loc(c) for c in NUMERIC_COLS]
        assert np.abs(X_tr[:, idx].mean(axis=0)).max() < 0.1

    def test_no_nan_after_scaling(self, scaled):
        X_tr, X_va, X_te, *_ = scaled
        for arr in (X_tr, X_va, X_te):
            assert not np.isnan(arr).any()

    def test_class_weights_favor_minority(self, splits):
        _, _, _, y_train, _, _ = splits
        w = get_class_weights(y_train.values)
        assert w[1] > w[0]


class TestOptimizers:

    def test_all_names_build(self):
        for name in OPTIMIZER_NAMES:
            assert get_optimizer(name) is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_optimizer('not_an_optimizer')

    def test_momentum_is_nonzero(self):
        opt = get_optimizer('momentum')
        assert float(opt.momentum) > 0


class TestModel:

    def test_builds(self):
        assert build_model(input_dim=30) is not None

    def test_output_is_single_sigmoid(self):
        model = build_model(input_dim=30)
        assert model.output_shape == (None, 1)

    def test_probabilities_in_range(self):
        model = build_model(input_dim=30)
        proba = model.predict(np.random.rand(16, 30), verbose=0)
        assert proba.min() >= 0.0 and proba.max() <= 1.0

    def test_accepts_each_optimizer(self):
        for name in OPTIMIZER_NAMES:
            model = build_model(input_dim=30, optimizer_name=name)
            assert model.optimizer is not None


class TestRegularizationAndMetrics:

    def test_dense_layers_have_l2(self):
        model = build_model(input_dim=30)
        dense = [l for l in model.layers if l.__class__.__name__ == 'Dense']
        # every hidden Dense layer should carry an L2 penalty
        hidden = [l for l in dense if l.name.startswith('dense_')]
        assert all(l.kernel_regularizer is not None for l in hidden)

    def test_tracks_auc_precision_recall(self):
        # After one training step the metric names are populated.
        model = build_model(input_dim=30)
        h = model.fit(np.random.rand(16, 30), np.random.randint(0, 2, 16),
                       epochs=1, batch_size=8, verbose=0)
        keys = ' '.join(h.history.keys()).lower()
        assert 'auc' in keys and 'precision' in keys and 'recall' in keys


class TestEvaluation:

    def test_metrics_present(self, small_model, scaled):
        _, _, X_te, _, _, y_te = scaled
        m = evaluate_model(small_model, X_te, y_te, name='adam')
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'loss']:
            assert key in m

    def test_metrics_in_unit_range(self, small_model, scaled):
        _, _, X_te, _, _, y_te = scaled
        m = evaluate_model(small_model, X_te, y_te, name='adam')
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            assert 0.0 <= m[key] <= 1.0

    def test_predictions_match_threshold(self, small_model, scaled):
        _, _, X_te, _, _, y_te = scaled
        m = evaluate_model(small_model, X_te, y_te, name='adam')
        rebuilt = (m['y_proba'] >= THRESHOLD).astype(int)
        assert np.array_equal(rebuilt, m['y_pred'])

    def test_prediction_count_matches(self, small_model, scaled):
        _, _, X_te, _, _, y_te = scaled
        m = evaluate_model(small_model, X_te, y_te, name='adam')
        assert len(m['y_pred']) == len(y_te)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))