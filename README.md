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


# Task 3: Put your brains into your PC - Multi Layer Perceptrons
   In this task we have learnt how neural networks work and maths behind the code whie understanding the essence of perceptrons, how perceptrons are related to neural networks too.
