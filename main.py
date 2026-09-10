import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def rmse(y, yh): return np.sqrt(np.mean((y - yh) ** 2))
def mae(y, yh):  return np.mean(np.abs(y - yh))
def r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - np.mean(y)) ** 2)
def acc(y, yh):  return np.mean(y == yh)
def scale(X):
    m, s = X.mean(0), X.std(0)
    s[s == 0] = 1
    return (X - m) / s, m, s

def split(X, y, test=0.2, seed=42):
    idx = np.random.RandomState(seed).permutation(len(X))
    k = int(len(X) * (1 - test))
    return X[idx[:k]], X[idx[k:]], y[idx[:k]], y[idx[k:]]
def linreg_gd(X, y, loss='mse', lr=0.05, iters=3000):
    m, n = X.shape
    w, b = np.zeros(n), 0.0
    hist = []
    for _ in range(iters):
        yh = X @ w + b
        r = yh - y
        if loss == 'mse':
            c = np.mean(r ** 2); g = 2 * r / m
        elif loss == 'mae':
            c = np.mean(np.abs(r)); g = np.sign(r) / m
        else:  # huber
            d = 1.0; a = np.abs(r)
            c = np.mean(np.where(a <= d, 0.5 * a**2, d * (a - 0.5 * d)))
            g = np.where(a <= d, r, d * np.sign(r)) / m
        hist.append(c)
        w -= lr * (X.T @ g)
        b -= lr * g.sum()
    return w, b, hist
def linreg_ols(X, y):
    Xb = np.c_[np.ones(len(X)), X]
    t = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
    return t[1:], t[0]
def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def logreg(X, y, lr=0.3, iters=3000):
    m, n = X.shape
    w, b = np.zeros(n), 0.0
    for _ in range(iters):
        p = sigmoid(X @ w + b)
        err = p - y
        w -= lr * (X.T @ err) / m
        b -= lr * err.sum() / m
    return w, b
def load_weather():
    df = pd.read_csv('data/weatherHistory.csv')
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df['hr'] = df['Formatted Date'].dt.hour
    df['mo'] = df['Formatted Date'].dt.month
    df['hs'] = np.sin(2*np.pi*df['hr']/24); df['hc'] = np.cos(2*np.pi*df['hr']/24)
    df['ms'] = np.sin(2*np.pi*df['mo']/12); df['mc'] = np.cos(2*np.pi*df['mo']/12)
    cols = ['Humidity','Wind Speed (km/h)','Wind Bearing (degrees)',
            'Visibility (km)','Pressure (millibars)','hs','hc','ms','mc']
    df = df[cols + ['Temperature (C)']].dropna()
    return df[cols].values, df['Temperature (C)'].values

def load_candy():
    df = pd.read_csv('data/candy-data.csv').drop(columns=['competitorname'])
    cols = [c for c in df.columns if c != 'winpercent']
    return df[cols].values, df['winpercent'].values
if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)

    print('\n' + '='*60)
    print('WEATHER - Temperature regression')
    print('='*60)
    X, y = load_weather()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, mu, sd = scale(Xtr)
    Xte = (Xte - mu) / sd

    for loss in ['mse', 'mae', 'huber']:
        w, b, _ = linreg_gd(Xtr, ytr, loss=loss)
        p = Xte @ w + b
        print(f'{loss:6s} | RMSE {rmse(yte,p):6.3f} | MAE {mae(yte,p):6.3f} | R2 {r2(yte,p):6.4f}')

    w, b = linreg_ols(Xtr, ytr)
    p = Xte @ w + b
    print(f'{"ols":6s} | RMSE {rmse(yte,p):6.3f} | MAE {mae(yte,p):6.3f} | R2 {r2(yte,p):6.4f}')
    print('\n--- outlier experiment (5% outliers added to train) ---')
    rng = np.random.RandomState(0)
    n_out = int(0.05 * len(ytr))
    idx = rng.choice(len(ytr), n_out, replace=False)
    ytr_noisy = ytr.copy()
    ytr_noisy[idx] += rng.choice([-1, 1], n_out) * 60.0

    results = {}
    for loss in ['mse', 'mae', 'huber']:
        w, b, _ = linreg_gd(Xtr, ytr_noisy, loss=loss, iters=4000)
        p = Xte @ w + b
        results[loss] = rmse(yte, p)
        print(f'{loss:6s} | clean test RMSE {results[loss]:6.3f}')

    plt.figure(figsize=(6, 4))
    plt.bar(list(results.keys()), list(results.values()),
            color=['tab:red', 'tab:green', 'tab:blue'])
    plt.ylabel('RMSE on clean test set')
    plt.title('Effect of 5% training outliers')
    plt.grid(True, axis='y'); plt.tight_layout()
    plt.savefig('results/loss_comparison.png', dpi=150); plt.close()
    print('\n' + '='*60)
    print('CANDY')
    print('='*60)
    X, y = load_candy()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, mu, sd = scale(Xtr)
    Xte = (Xte - mu) / sd

    w, b = linreg_ols(Xtr, ytr)
    p = Xte @ w + b
    print(f'winpercent regression | RMSE {rmse(yte,p):6.3f} | R2 {r2(yte,p):6.4f}')

    ybin = (y > np.median(y)).astype(int)
    Xtr, Xte, ytr, yte = split(X, ybin)
    Xtr, mu, sd = scale(Xtr)
    Xte = (Xte - mu) / sd
    w, b = logreg(Xtr, ytr)
    p = (sigmoid(Xte @ w + b) >= 0.5).astype(int)
    print(f'top-half classification | accuracy {acc(yte,p):.4f}')

    print('\n--- overfitting demo (polynomial degree sweep) ---')
    rng = np.random.RandomState(0)
    xs = np.linspace(-3, 3, 100).reshape(-1, 1)
    ys = np.sin(xs.ravel()) + rng.normal(0, 0.15, 100)
    Xtr_p, Xte_p = xs[:70], xs[70:]
    ytr_p, yte_p = ys[:70], ys[70:]

    def poly(X, d): return np.hstack([X ** k for k in range(1, d + 1)])
    degs = [1, 3, 5, 10, 15]
    tr_r2, te_r2 = [], []
    for d in degs:
        w, b = linreg_ols(poly(Xtr_p, d), ytr_p)
        tr_r2.append(r2(ytr_p, poly(Xtr_p, d) @ w + b))
        te_r2.append(r2(yte_p, poly(Xte_p, d) @ w + b))
        print(f'deg={d:2d} | train R2 {tr_r2[-1]:.3f} | test R2 {te_r2[-1]:.3f}')

    plt.figure(figsize=(6, 4))
    plt.plot(degs, tr_r2, 'o-', label='train R2')
    plt.plot(degs, te_r2, 's-', label='test R2')
    plt.xlabel('polynomial degree'); plt.ylabel('R2'); plt.legend()
    plt.title('Overfitting: train R2 rises, test R2 collapses')
    plt.grid(True); plt.tight_layout()
    plt.savefig('results/overfitting.png', dpi=150); plt.close()

    print('\nSaved plots to results/')
    print('Done.')