import os, tensorflow as tf
from sklearn.metrics import recall_score, precision_score
from preprocessing import run_preprocessing

OUT = 'outputs'
data = run_preprocessing(os.path.join('data', 'teleconnect.csv'),
                         scaler_save_path=os.path.join(OUT, 'scaler.pkl'))
model = tf.keras.models.load_model(os.path.join(OUT, 'best_model.keras'))

X_test, y_test, feat = data['X_test'], data['y_test'], data['feature_names']
pred = (model.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)

def report(mask, label):
    yt, yp = y_test[mask], pred[mask]
    print(f"{label:12s} n={int(mask.sum()):4d}  "
          f"recall={recall_score(yt, yp, zero_division=0):.3f}  "
          f"precision={precision_score(yt, yp, zero_division=0):.3f}")

g = X_test[:, feat.index('gender')]          # 1 = Male, 0 = Female
s = X_test[:, feat.index('SeniorCitizen')]   # 1 = Senior
report(g == 1, 'Male'); report(g == 0, 'Female')
report(s == 1, 'Senior'); report(s == 0, 'Non-senior')