""" Analysis function developed by the lab """
import numpy as np
from sklearn.model_selection import KFold

"""
Perform k-fold cross-validation for regression with different ranks of prediction matrix.
 - Implements ridge regression with optional reduced rank approximation.
 - Handles optional z-score normalization and per-factor performance metrics.

    Args:
        X (np.ndarray): Data matrix of predictors (time * components).
        Y (np.ndarray): Data matrix of target to predict.
        rank_iterator (iterable): List or range of ranks to be tested.
        sigma (float): Ridge regularization parameter.
        nfolds (int): Number of folds for cross-validation.
        perf (bool, optional): If True, calculate performance metrics per factor. Default is True.
        zscore (bool, optional): If True, z-score normalize the data before processing. Default is False.
    
    Returns:
        tuple:
            - np.ndarray: Mean training R² values for each rank.
            - np.ndarray: Mean testing R² values for each rank.
            - np.ndarray: Mean performance R² values per factor.
            - np.ndarray: Full predictions for the Y matrix.
"""
def cv_regression(X, Y, rank_iterator=(0,), sigma=0, nfolds=5, perf=True, zscore=False):
    # Initialize KFold cross-validation
    kfolds = KFold(n_splits=nfolds, random_state=None, shuffle=True)
    taxis = range(X.shape[0])

    # Initialize arrays to store R² values for training and testing
    train_r2 = np.full([nfolds, len(rank_iterator)], np.nan)
    test_r2 = np.full([nfolds, len(rank_iterator)], np.nan)
    perf_r2 = np.full([nfolds, Y.shape[1]], np.nan)

    # Optionally z-score normalize the data
    if zscore:
        X = np.divide(X - np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True))
        Y = np.divide(Y - np.mean(Y, axis=0, keepdims=True), np.std(Y, axis=0, keepdims=True))

    # Initialize an array to store the full predictions
    Yfull_prediction = np.full(Y.shape, np.nan)

    # Perform k-fold cross-validation
    for ifold, (train, test) in enumerate(kfolds.split(taxis)):
        # Split data into training and testing sets
        Xtrainf = X[train, :]
        Ytrainf = Y[train, :]
        Xtestf = X[test, :]
        Ytestf = Y[test, :]

        # Optionally z-score normalize the training and testing data
        if zscore:
            Xtrainf = np.divide(Xtrainf - np.mean(Xtrainf, axis=0, keepdims=True),
                                np.std(Xtrainf, axis=0, keepdims=True))
            Ytrainf = np.divide(Ytrainf - np.mean(Ytrainf, axis=0, keepdims=True),
                                np.std(Ytrainf, axis=0, keepdims=True))
            Xtestf = np.divide(Xtestf - np.mean(Xtestf, axis=0, keepdims=True), np.std(Xtestf, axis=0, keepdims=True))
            Ytestf = np.divide(Ytestf - np.mean(Ytestf, axis=0, keepdims=True), np.std(Ytestf, axis=0, keepdims=True))

        # Center the training and testing data
        Ytrainf = Ytrainf - np.mean(Ytrainf, axis=0, keepdims=True)
        Ytestf = Ytestf - np.mean(Ytestf, axis=0, keepdims=True)
        Xtrainf = Xtrainf - np.mean(Xtrainf, axis=0, keepdims=True)
        Xtestf = Xtestf - np.mean(Xtestf, axis=0, keepdims=True)

        # Perform ridge regression and obtain the reduced rank regression components
        V, B_OLS, Ytrain, Xtrain = reduced_reg(Xtrainf, Ytrainf, sigma)
        V = V.transpose()

        # Iterate over each rank in rank_iterator
        for irank, rank in enumerate(rank_iterator):
            B_r = B_OLS
            if rank > 0:
                Vr = V[:, :int(rank)]
                B_r = np.dot(B_OLS, np.dot(Vr, Vr.T))

            # Calculate training error and R² value
            train_err = Ytrainf - np.dot(Xtrain, B_r)
            train_err = train_err.flatten()
            train_mse = np.mean(np.power(train_err, 2))
            ss = np.mean(np.power(Ytrainf.flatten(), 2))
            train_r2[ifold, irank] = 1 - train_mse / ss

            # Calculate testing error and R² value
            Ytestpred = np.dot(Xtestf, B_r)
            Ytestpred = Ytestpred - np.mean(Ytestpred, axis=0, keepdims=True)
            test_err = Ytestf - Ytestpred
            test_mse = np.mean(np.power(test_err.flatten(), 2))
            test_ss = np.mean(np.power(Ytestf.flatten(), 2))
            test_r2[ifold, irank] = 1 - test_mse / test_ss
            Yfull_prediction[test, :] = Ytestpred

            # Calculate performance R² values per factor if required
            if rank == 0 and perf:  # This only works with full rank
                perf_mse = np.mean(test_err ** 2, axis=0)  # vector, one mse value per factor
                perf_sst = np.mean((Ytestf - np.mean(Ytestf, axis=0)) ** 2, axis=0)
                perf_r2[ifold, :] = 1 - (perf_mse / perf_sst)

    return np.mean(train_r2, axis=0), np.mean(test_r2, axis=0).item(), np.mean(perf_r2, axis=0), Yfull_prediction


"""
Perform ridge regression and return the reduced rank regression components.

    Args:
        X (np.ndarray): Data matrix of predictors (time * components).
        Y (np.ndarray): Data matrix of target to predict.
        sigma (float): Ridge regularization parameter.

    Returns:
        tuple: 
            - np.ndarray: Right singular vectors (V) of the Y_OLS matrix.
            - np.ndarray: Regression coefficients (B_OLS).
            - np.ndarray: Centered Y matrix.
            - np.ndarray: Centered X matrix.
"""
def reduced_reg(X, Y, sigma):
    # Center the X and Y matrices
    mX = np.mean(X, axis=0, keepdims=True)
    mY = np.mean(Y, axis=0, keepdims=True)
    X = X - mX
    Y = Y - mY

    # Compute the covariance matrices
    CXX = np.dot(X.T, X) + sigma ** 2 * np.identity(np.size(X, 1))
    CXY = np.dot(X.T, Y)

    # Perform ridge regression to obtain the regression coefficients
    B_OLS = np.dot(np.linalg.pinv(CXX), CXY)

    # Obtain the predicted values for Y
    Y_OLS = np.dot(X, B_OLS)

    # Perform singular value decomposition (SVD) on the predicted Y matrix
    _U, _S, V = np.linalg.svd(Y_OLS, full_matrices=False)

    return V, B_OLS, Y, X
