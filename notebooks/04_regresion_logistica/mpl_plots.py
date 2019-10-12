import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors.classification import KNeighborsClassifier


def safe_margin(val, low=True, pct: float = 0.05):
    low_pct, high_pct = 1 - pct, 1 + pct
    func = min if low else max
    return func(val * low_pct, val * high_pct)


def safe_bounds(array, pct: float = 0.05):
    low_x, high_x = array.min(), array.max()
    low_x = safe_margin(low_x, pct=pct)
    high_x = safe_margin(high_x, pct=pct, low=False)
    return low_x, high_x


def example_meshgrid(X, n_bins=100, low_th: float = 0.95, high_th: float = 1.05):
    low_x, high_x = X[:, 0].min(), X[:, 0].max()
    low_y, high_y = X[:, 1].min(), X[:, 1].max()
    low_x = safe_margin(low_x)
    low_y = safe_margin(low_y)
    high_x = safe_margin(high_x, False)
    high_y = safe_margin(high_y, False)
    xs = np.linspace(low_x, high_x, n_bins)
    ys = np.linspace(low_y, high_y, n_bins)
    return np.meshgrid(xs, ys)


def predict_grid(model, X):
    x_grid, y_grid = example_meshgrid(X)
    grid = np.c_[x_grid.ravel(), y_grid.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(x_grid.shape)
    return probs, x_grid, y_grid


def plot_dataset_2d(X_train, y_train, X_test, y_test):
    s = 400
    plt.figure(figsize=(10, 5))
    idx = y_train == 0
    idx_test = y_test == 0
    plt.scatter(X_train[idx, 0], X_train[idx, 1], label="Clase 0 train", color="b", s=s, alpha=0.5)
    plt.scatter(
        X_train[~idx, 0], X_train[~idx, 1], label="Clase 1 train", color="r", s=s, alpha=0.5
    )
    plt.scatter(
        X_test[idx_test, 0],
        X_test[idx_test, 1],
        label="Clase 0 test",
        color="b",
        s=s,
        alpha=0.5,
        marker="s",
    )
    plt.scatter(
        X_test[~idx_test, 0],
        X_test[~idx_test, 1],
        label="Clase 1 test",
        color="r",
        s=s,
        alpha=0.5,
        marker="s",
    )
    plt.title("Dataset de ejemplo")
    plt.xlim((7.5, 12.3))
    plt.ylim((-1.5, 6))
    plt.legend(loc="best", labelspacing=1)


def plot_classes(X, y, s: int=100, probs=None, model=None, cmap="bwr", *args, **kwargs):
    if model is not None:
        try:
            probs = model.predict_proba(X)[:, 1]
        except:
            probs = model.predict_proba(X)
    data = pd.DataFrame(
        {"x": X[:, 0], "y": X[:, 1], "target": y, "prob": probs}
    )
    return data.plot.scatter(x="x", y="y", c="target",
                             cmap=cmap, s=s, colorbar=False,
                             *args, **kwargs)


def plot_boundary(model, min_x, max_x, color="#cfcb02", *args, **kwargs):
    theta = np.concatenate(
        [model.intercept_ if isinstance(model.intercept_, np.ndarray) else [model.intercept_],
         model.coef_[0]]
    )  # getting the x co-ordinates of the decision boundary
    plot_x = np.array([min_x, max_x])
    # getting corresponding y co-ordinates of the decision boundary
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])  # Plotting the Single Line Decision
    # Boundary
    data = pd.DataFrame({"x": plot_x, "y": plot_y})
    return data.plot(color=color, x="x", y="y", *args, **kwargs)


def plot_model_output(model, X, y, title=None, s=300):
    fig, ax = plt.subplots(figsize=(8, 8))
    ps, XX, YY = predict_grid(model, X)
    _ = ax.pcolormesh(XX, YY, ps, cmap=plt.cm.YlGnBu)
    min_x, max_x = safe_bounds(X[:, 0], 0.01)
    ax = plot_boundary(model=model, min_x=min_x, max_x=max_x, ax=ax, linewidth=4, alpha=0.9,
                       label="Frontera de decisión")
    ax = plot_classes(X, y, model=model, ax=ax, alpha=0.8, s=s)
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=22)
    plt.xlim(min_x, max_x)
    plt.ylim(*safe_bounds(X[:, 1], 0.01))
    return fig, ax


def plot_decision_boundaries(
    X_train,
    y_train,
    y_pred_train,
    X_test,
    y_test,
    y_pred_test,
    resolution: int = 100,
    embedding=None,
    figsize=(9, 8),
    cmap="viridis",
    title: str="Decision boundaries",
    s=200,
):
    import umap
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    y_pred = np.concatenate([y_pred_train, y_pred_test])

    if embedding is None:
        try:
            embedding = umap.UMAP(n_components=2, random_state=160290).fit_transform(X)
        except ImportError:
            from sklearn.manifold import TSNE

            embedding = TSNE(n_components=2, random_state=160290).fit_transform(X)
    x_min, x_max = safe_bounds(embedding[:, 0])
    y_min, y_max = safe_bounds(embedding[:, 1])
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(embedding, y_pred)
    voronoi_bg = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoi_bg = voronoi_bg.reshape((resolution, resolution))
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(xx, yy, voronoi_bg,cmap=cmap, alpha=0.1)
    emb_train = embedding[: len(y_train)]
    data = pd.DataFrame(
        {"x": emb_train[:, 0], "y":emb_train[:, 1], "target": y_train}
    )
    data.plot.scatter(x="x", y="y", c="target",
                      cmap=cmap, s=s, colorbar=False,
                      ax=ax, alpha=0.7, label="train set")
    emb_test = embedding[len(y_train):]
    data = pd.DataFrame(
        {"x": emb_test[:, 0], "y":emb_test[:, 1], "target": y_test}
    )
    data.plot.scatter(x="x", y="y", c="target",
                      cmap=cmap, s=s, colorbar=False,
                      ax=ax, alpha=0.7, marker="s", label="test set")
    errors = y_pred != y
    failed_points = ax.scatter(embedding[errors, 0], embedding[errors, 1],
                               c="red", s=50, alpha=0.9, label="errors")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=22)
    return fig, ax


def plot_confusion_matrix(
        y_test,
        y_pred,
        target_names: list = None,
        cmap: str = "YlGnBu",
        figsize=(9, 8),
        title: str = "Confusion matrix",
        annot_fontsize=22,
        title_fontsize=22,
        label_fontsize=22,
        ticks_fontsize=14,
):
    fig, ax = plt.subplots(figsize=figsize)
    if target_names is not None:
        df = pd.DataFrame(confusion_matrix(y_test, y_pred), index=target_names,
                          columns=target_names)
    else:
        df = pd.DataFrame(confusion_matrix(y_test, y_pred))

    sns.heatmap(df, cmap=cmap, annot=True, ax=ax, annot_kws={"fontsize": annot_fontsize})
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel("Predicted label", fontsize=label_fontsize)
    plt.ylabel("Actual label", fontsize=label_fontsize)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_fontsize)

    return fig, ax