linreg_ridge_1 = Ridge(alpha=1).fit(X_train, y_train)
print(linreg_ridge_1.score(X_train, y_train))
print(linreg_ridge_1.score(X_test, y_test))

linreg_ridge_01 = Ridge(alpha=0.1).fit(X_train, y_train)
print(linreg_ridge_01.score(X_train, y_train))
print(linreg_ridge_01.score(X_test, y_test))

linreg_ridge_001 = Ridge(alpha=0.01).fit(X_train, y_train)
print(linreg_ridge_001.score(X_train, y_train))
print(linreg_ridge_001.score(X_test, y_test))