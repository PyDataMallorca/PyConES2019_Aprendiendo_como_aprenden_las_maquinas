linreg_lasso_0005 = Lasso(alpha=0.005, max_iter=100_000).fit(X_train, y_train)
print(linreg_lasso_0005.score(X_train, y_train))
print(linreg_lasso_0005.score(X_test, y_test))
print(np.sum(linreg_lasso_0005.coef_ != 0))

linreg_lasso_005 = Lasso(alpha=0.5, max_iter=100_000).fit(X_train, y_train)
print(linreg_lasso_005.score(X_train, y_train))
print(linreg_lasso_005.score(X_test, y_test))
print(np.sum(linreg_lasso_005.coef_ != 0))