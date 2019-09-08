from sklearn.linear_model import RidgeCV

alphas = np.arange(0.005, 1, 0.005)
linreg_ridge_cv = RidgeCV(alphas).fit(X_train, y_train)
print(linreg_ridge_cv.alpha_)