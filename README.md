# Task 1: Lost in Hubris - PROJECT MANAS

Name: Sai Akshay Kumar

Linear and Logistic Regression from scratch using only numpy. No sklearn.

Datasets:
- Szeged weather -> predict Temperature
- Halloween candy -> predict winpercent + top-half classification

Run:
    pip install numpy pandas matplotlib
    python main.py

## Why MSE isn't suitable
MSE squares the error so outliers dominate. tested this by adding 5% outliers
(+/- 60 C) to the weather training set and re-evaluating on the clean test set.

    mse    -> 4.003
    mae    -> 4.020
    huber  -> 4.018
The important lesson: MSE's gradient grows linearly with the error, so one
huge outlier produces a huge update. MAE's gradient is always +/- 1, so outliers
have bounded influence. Huber is a blend - quadratic when errors are small,
linear when large - so it's robust without losing the smooth gradient.

## Output function to check it's process

80/20 train/test split on everything. All numbers below are on the test set.

Weather (predicting Temperature from humidity, wind, pressure, visibility, hour, month):

    mse    | RMSE  4.002 | MAE  3.169 | R2  0.8256
    mae    | RMSE  4.019 | MAE  3.159 | R2  0.8241
    huber  | RMSE  4.018 | MAE  3.159 | R2  0.8242
    ols    | RMSE  4.002 | MAE  3.169 | R2  0.8256

OLS and MSE give identical results - expected, since OLS is the closed-form
solution to MSE.

Candy:

    winpercent regression   | RMSE 12.238 | R2 0.4647
    top-half classification | accuracy 0.7647

To check overfitting I fit polynomials of increasing degree to a noisy sine
wave. Train R2 stays near 0.94 while test R2 collapses into the billions:

    deg= 1 | train R2 0.453 | test R2 -1.624
    deg= 3 | train R2 0.942 | test R2 -1.937
    deg= 5 | train R2 0.944 | test R2 -20.077
    deg=10 | train R2 0.946 | test R2 -7562837.329
    deg=15 | train R2 0.943 | test R2 -20187202275.673

# Task 4: Rolling in the Deep

This repo has my implementation for Task 4 of the AI taskphase. The goal was to build a baseline MLP on the CIFAR-10 dataset[cite: 4] and compare how different optimisers (written from scratch) and regularisation methods affect training and validation loss.

### Model Architecture
I kept the architecture fixed across all runs so the comparisons are fair:
- Input: Flattened CIFAR-10 images (`32x32x3` -> `3072`)
- Layer 1: `Linear(3072, 512)` -> ReLU
- Layer 2: `Linear(512, 256)` -> ReLU
- Output: `Linear(256, 10)` with Cross-Entropy Loss

For the BatchNorm and Dropout experiments, I toggled `BatchNorm1d` (before ReLU) and `Dropout(0.3)` (after ReLU) inside the same base class. Everything was trained for 10 epochs with a batch size of 128 using the `mps` backend.

### What I Implemented

1. **Custom Optimisers (From Scratch)**
   Instead of using `torch.optim` for everything, I wrote custom classes iterating over model parameters inside `torch.no_grad()` blocks:
   - `ScratchMomentumOptimizer`: Tracks velocity buffers for each parameter (`lr=0.01`, `momentum=0.9`).
   - `ScratchAdamOptimizer`: Tracks first (`m`) and second (`v`) moment buffers along with step-wise bias correction (`lr=0.001`, `betas=(0.9, 0.999)`, `eps=1e-8`).

2. **Regularisation & Scheduling Runs**
   - **Baseline:** Plain MLP with standard `optim.SGD` (`lr=0.01`).
   - **BatchNorm:** Added `BatchNorm1d` layers to help with internal covariate shift and gradient flow.
   - **Dropout:** Added `p=0.3` dropout to reduce overfitting.
   - **Cosine Annealing:** Started with a higher learning rate (`0.1`) on SGD and decayed it over the 10 epochs using `CosineAnnealingLR`.

### Observations from the Loss Curves

- **Plain SGD vs Momentum & Adam:** Plain SGD at `lr=0.01` learns pretty slowly. Adding custom Momentum speeds up convergence a lot in the first few epochs. Scratch Adam drops the loss the fastest early on, though it starts plateauing/overfitting slightly towards epoch 8-10 since MLPs memorize pixel positions easily on CIFAR-10.
- **BatchNorm:** Made a massive difference even with basic SGD. The validation loss dropped much faster than the
