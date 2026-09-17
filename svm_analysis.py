"df = df.dropna()""Task 2a: SVM with Linear, Polynomial, RBF kernels on two datasets."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

RANDOM_STATE = 42
FIGS = Path("results/figures"); FIGS.mkdir(parents=True, exist_ok=True)
METRICS = Path("results/metrics"); METRICS.mkdir(parents=True, exist_ok=True)


def run_dataset(name, path, target_col, class_weight=None):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print("columns:", df.columns.tolist())

    if target_col not in df.columns:
        target_col = df.columns[-1]
        print(f"  using last column as target: {target_col}")
    df = df.dropna()
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    print(f"train {Xtr.shape}, pos rate {ytr.mean():.3f}")

    results = {}
    for kernel in ["linear", "poly", "rbf"]:
        print(f"  training {kernel} ...")
        m = SVC(kernel=kernel, C=1.0, gamma="scale", degree=3,
                probability=True, class_weight=class_weight,
                random_state=RANDOM_STATE).fit(Xtr, ytr)
        yp = m.predict(Xte)
        yprob = m.predict_proba(Xte)[:, 1]
        results[kernel] = {
            "model": m, "y_pred": yp, "y_prob": yprob,
            "accuracy":  accuracy_score(yte, yp),
            "precision": precision_score(yte, yp, zero_division=0),
            "recall":    recall_score(yte, yp, zero_division=0),
            "f1":        f1_score(yte, yp, zero_division=0),
            "roc_auc":   roc_auc_score(yte, yprob),
            "n_sv":      len(m.support_),
        }

    table = pd.DataFrame({
        k: {"accuracy": r["accuracy"], "precision": r["precision"],
            "recall": r["recall"], "f1": r["f1"],
            "roc_auc": r["roc_auc"], "n_sv": r["n_sv"]}
        for k, r in results.items()
    }).T
    print(table.round(3))

    plt.figure(figsize=(6, 5))
    for k, r in results.items():
        fpr, tpr, _ = roc_curve(yte, r["y_prob"])
        plt.plot(fpr, tpr, label=f"{k} (AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGS / f"{name.lower()}_roc.png", dpi=120); plt.close()

    best = max(results, key=lambda k: results[k]["f1"])
    cm = confusion_matrix(yte, results[best]["y_pred"])
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title(f"{name} best: {best}"); plt.tight_layout()
    plt.savefig(FIGS / f"{name.lower()}_cm.png", dpi=120); plt.close()

    return table, results, Xtr, ytr


tab_p, res_p, Xp, yp = run_dataset(
    "Pulsar", "data/pulsar/pulsar_data_train.csv",
    "target_class", class_weight="balanced")

tab_g, res_g, Xg, yg = run_dataset(
    "Genomic", "data/genomic/gene_expression.csv",
    "cancer", class_weight=None)

if Xg.shape[1] == 2:
    import numpy as np
    h = 0.02
    x_min, x_max = Xg[:, 0].min()-1, Xg[:, 0].max()+1
    y_min, y_max = Xg[:, 1].min()-1, Xg[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, k in zip(axes, ["linear", "poly", "rbf"]):
        m = res_g[k]["model"]
        Z = m.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(Xg[:, 0], Xg[:, 1], c=yg, cmap="coolwarm",
                   edgecolors="k", s=25)
        ax.set_title(f"{k} kernel")
    plt.suptitle("Genomic - Decision Boundaries"); plt.tight_layout()
    plt.savefig(FIGS / "genomic_boundaries.png", dpi=120); plt.close()


def to_df(results, name):
    return pd.DataFrame([
        {"dataset": name, "kernel": k,
         "accuracy": r["accuracy"], "precision": r["precision"],
         "recall": r["recall"], "f1": r["f1"],
         "roc_auc": r["roc_auc"], "n_support_vectors": r["n_sv"]}
        for k, r in results.items()])

combined = pd.concat([to_df(res_p, "pulsar"),
                      to_df(res_g, "genomic")], ignore_index=True)
combined.to_csv(METRICS / "kernel_comparison.csv", index=False)
print("\nSaved results/metrics/kernel_comparison.csv")
print(combined.round(3))
