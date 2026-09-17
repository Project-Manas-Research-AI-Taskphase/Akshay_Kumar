# Task 2a — Kernels but not from Popcorns
SVM classification with Linear,  Polynomial (degree 3), and RBF kernels on two datasets.

 Datasets
- Pulsar Star — 8 features, 12,528 samples, ~9% positive (imbalanced)
- Genomic Cancer — 2 gene features, 3,000 samples, balanced

Results

 nPulsar Star
| Kernel | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Linear | 0.967 | 0.771 | 0.912 | 0.836 | 0.976|
| Poly-3 | 0.969| 0.812 | 0.865 | 0.838| 0.969 |
| RBF    | 0.968 | 0.781 | 0.900 | 0.836 | 0.970 |

  Genomic Cancer
| Kernel | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Linear | 0.873 | 0.866 | 0.883 | 0.875 | 0.947 |
| Poly-3 | 0.865 | 0.874 | 0.853 | 0.863 | 0.959 |
| RBF | 0.947| 0.932| 0.963 | 0.948 | 0.986 |


Pulsar: Basically three kernels had their own better performances like how polynomial has best precision while ref had best recall and linear has best ROC-AUC where all are are linearly separable

Genomic:RBF clearly won on every metric , using only 488 support vectors. W
the classes are not linearly separable, so rbf’s implicit higher-dimensional
mapping captures the curved boundary the linear kernel cannot.

Imbalance:Pulsar is ~9% positive, so `class_weight='balanced'` was used.
Accuracy is misleading here — F1 and ROC-AUC are reported alongside.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/svm_analysis.py
cd ~/task2-svm-kernels
pwd
ls
## Run
python src/svm_analysis.py

## Outputs
results/metrics/kernel_comparison.csv
results/figures/*.png

## Data
Download from Kaggle:
https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate
https://www.kaggle.com/datasets/brsahan/genomic-data-for-cancer
