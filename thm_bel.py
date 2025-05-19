import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from properscoring import crps_ensemble

# =========================
# Evaluation Metrics
# =========================
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / (y_true + 1e-6)) ** 2))

def calculate_crps(y_true, samples):
    # samples: shape (n_samples, n_points, dim)
    ensemble = np.moveaxis(samples, 0, 1)  # -> (n_points, n_samples, dim)
    crps_vals = []
    for d in range(ensemble.shape[-1]):
        crps_vals.append(np.mean([crps_ensemble(ensemble[i,:,d], y_true[i,d]) for i in range(len(y_true))]))
    return np.mean(crps_vals), np.std(crps_vals)

# =========================
# MDN Pipeline
# =========================
def run_mdn(data_path, target_path, time_steps=250, horizon=50, n_components=6, epochs=100, batch_size=64):
    # Load
    data = pd.read_pickle(data_path).values
    data2 = pd.read_pickle(target_path).values
    X = data[:, :time_steps]
    y = data2[:, :horizon]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    X_train_s, X_test_s = scaler_X.transform(X_train), scaler_X.transform(X_test)
    y_train_s, y_test_s = scaler_y.transform(y_train), scaler_y.transform(y_test)

    # PCA
    pca_X = PCA(n_components=6).fit(X_train_s)
    pca_y = PCA(n_components=2).fit(y_train_s)
    X_train_pca, X_test_pca = pca_X.transform(X_train_s), pca_X.transform(X_test_s)
    y_train_pca, y_test_pca = pca_y.transform(y_train_s), pca_y.transform(y_test_s)

    # Model
    inputs = tf.keras.Input(shape=X_train_pca.shape[1], name='input')
    h = tf.keras.layers.Dense(128, activation='relu')(inputs)
    h = tf.keras.layers.Dropout(0.3)(h)
    h = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(h)
    h = tf.keras.layers.Dropout(0.3)(h)
    h = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(h)
    params = tf.keras.layers.Dense(n_components * 3 * pca_y.n_components)(h)
    mdn = tfp.layers.MixtureNormal(num_components=n_components, event_shape=[pca_y.n_components])(params)
    model = tf.keras.Model(inputs, mdn)
    model.compile(loss=lambda y, rv: -rv.log_prob(y), optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))

    # Train
    history = model.fit(
        X_train_pca, y_train_pca,
        validation_data=(X_test_pca, y_test_pca),
        epochs=epochs, batch_size=batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    # Predict
    dist = model(X_test_pca)
    samples = dist.sample(100).numpy()  # (100, n_test, dim)
    # Inverse transforms
    reconst = np.stack([pca_y.inverse_transform(s) for s in samples], axis=0)
    reconst = scaler_y.inverse_transform(reconst.reshape(-1, reconst.shape[-1])).reshape(reconst.shape)

    # Metrics
    mean_pred = np.mean(reconst, axis=0)
    rmse = calculate_rmse(y_test, mean_pred)
    rmspe = calculate_rmspe(y_test, mean_pred)
    crps_mean, crps_std = calculate_crps(y_test, reconst)

    return {
        'rmse': rmse,
        'rmspe': rmspe,
        'crps_mean': crps_mean,
        'crps_std': crps_std
    }

# =========================
# BEL Pipeline
# =========================
def run_bel(x_path, y_path, time_steps=50, horizon=50, n_posts=100):
    from skbel import BEL

    # Load
    X = pd.read_pickle(x_path).values
    Y = pd.read_pickle(y_path).values
    x = X[:, :time_steps]
    y = Y[:, :horizon]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale
    scaler_x = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train)
    x_train_s, x_test_s = scaler_x.transform(x_train), scaler_x.transform(x_test)
    y_train_s, y_test_s = scaler_y.transform(y_train), scaler_y.transform(y_test)

    # Model
    bel = BEL(
        mode='kde',
        X_pre_processing=Pipeline([('pca', PCA(n_components=6))]),
        Y_pre_processing=Pipeline([('pca', PCA(n_components=2))]),
        regression_model=CCA(n_components=2),
        n_comp_cca=2
    )
    bel_model, cca_corrs = bel.fit(x_train_s, y_train_s)

    # Predict
    y_pred_s, x_obs_c, cca_samples = bel.predict(
        X_obs=x_test_s, n_posts=n_posts, mode='kde', return_cca=True
    )
    # Inverse transforms
    recon = np.stack([pca.inverse_transform(s) for s, pca in zip(cca_samples, [Pipeline([('cca', bel_model), ('inv_pca', bel.Y_pre_processing.named_steps['pca'])])])], axis=0)
    reconst = scaler_y.inverse_transform(recon.reshape(-1, recon.shape[-1])).reshape(recon.shape)

    # Metrics
    mean_pred = np.mean(reconst, axis=0)
    rmse = calculate_rmse(y_test, mean_pred)
    rmspe = calculate_rmspe(y_test, mean_pred)
    crps_mean, crps_std = calculate_crps(y_test, reconst)

    return {
        'rmse': rmse,
        'rmspe': rmspe,
        'crps_mean': crps_mean,
        'crps_std': crps_std
    }

# =========================
# Main Entry
# =========================
if __name__ == '__main__':
    mdn_results = run_mdn('data_wt.pkl', 'data_ds.pkl')
    bel_results = run_bel('data_gt.pkl', 'data_gs.pkl')

    print("MDN Results:")
    print(mdn_results)
    print("\nBEL Results:")
    print(bel_results)
