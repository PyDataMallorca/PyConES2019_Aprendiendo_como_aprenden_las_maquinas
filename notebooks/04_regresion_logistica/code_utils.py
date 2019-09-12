import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.streamz
import hvplot
import hvplot.pandas
from holoviews import streams
import streamz
from streamz.dataframe import DataFrame as StreamzDataFrame
from bokeh.models import HoverTool
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.manifold.t_sne import TSNE
from panel import widgets
import panel as pn


def safe_margin(val, low=True, pct: float = 0.05):
    low_pct, high_pct = 1 - pct, 1 + pct
    func = min if low else max
    return func(val * low_pct, val * high_pct)


def safe_bounds(array, pct: float = 0.05):
    low_x, high_x = array.min(), array.max()
    low_x = safe_margin(low_x, pct=pct)
    high_x = safe_margin(high_x, pct=pct, low=False)
    return low_x, high_x


def plot_confussion_matrix(
    y_test,
    y_pred,
    target_names: list = None,
    cmap: str = "YlGnBu",
    width=500,
    height: int = 400,
    title: str = "Confusion matrix",
    normalize: bool = False,
):
    value_label = "examples"
    target_label = "true_label"
    pred_label = "predicted_label"
    label_color = "color"

    def melt_distances_to_heatmap(distances: pd.DataFrame) -> pd.DataFrame:
        dist_melt = pd.melt(distances.reset_index(), value_name=value_label, id_vars="index")
        dist_melt = dist_melt.rename(columns={"index": target_label, "variable": pred_label})
        dist_melt[target_label] = pd.Categorical(dist_melt[target_label])
        dist_melt[pred_label] = pd.Categorical(dist_melt[pred_label])
        coords = dist_melt.copy()
        coords[target_label] = dist_melt[target_label].values.codes
        coords[pred_label] = dist_melt[pred_label].values.codes
        return coords[[pred_label, target_label, value_label]]

    conf_matrix = confusion_matrix(y_test, y_pred)
    if normalize:
        conf_matrix = np.round(
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis], 3
        )
    # Adjust label color to make them readable when displayed on top of any colormap
    df = melt_distances_to_heatmap(pd.DataFrame(conf_matrix))
    mean = df[value_label].mean()
    df[label_color] = -df[value_label].apply(lambda x: int(x > mean))
    if target_names is not None:
        df[target_label] = df[target_label].apply(lambda x: target_names[x])
        df[pred_label] = df[pred_label].apply(lambda x: target_names[x])
    true_label_name = "Actual label"
    pred_label_name = "Predicted label"

    tooltip = [
        (true_label_name, "@{%s}" % target_label),
        (pred_label_name, "@{%s}" % pred_label),
        ("Examples", "@{%s}" % value_label),
    ]
    hover = HoverTool(tooltips=tooltip)
    heatmap = hv.HeatMap(df, kdims=[pred_label, target_label])
    heatmap.opts(title=title, colorbar=True, cmap=cmap, width=width, height=height, tools=[hover])
    labeled = heatmap * hv.Labels(heatmap).opts(text_color=label_color, cmap=cmap)
    return labeled.options(xlabel=pred_label_name, ylabel=true_label_name, invert_yaxis=True)


def plot_decision_boundaries(X, y, y_pred, resolution: int = 100):
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

    mesh = hv.QuadMesh((xx, yy, voronoi_bg)).opts(cmap="viridis")
    points = hv.Scatter(
        {"x": embedding[:, 0], "y": embedding[:, 1], "pred": y_pred, "class": y},
        kdims=["x", "y"],
        vdims=["pred", "class"],
    )
    errors = y_pred != y
    failed_points = hv.Scatter(
        {"x": embedding[errors, 0], "y": embedding[errors, 1]}, kdims=["x", "y"]
    ).opts(color="red", size=5, alpha=0.9)

    points = points.opts(
        color="pred", cmap="viridis", line_color="black", size=10, alpha=0.8, tools=["hover"]
    )
    plot = mesh * points * failed_points
    plot = plot.opts(
        xaxis=None, yaxis=None, width=500, height=450, title="Decision boundaries on TSNE"
    )
    return plot


def plot_classification_report(y, y_pred, **kwargs):
    report = classification_report(y, y_pred, output_dict=True, **kwargs)
    df = pd.DataFrame(report).applymap(lambda x: "{:.2f}".format(x))
    df = df.T.reset_index()
    return hv.Table(df).opts(title="Classification report")


def plot_feature_importances(model, target_names=None, feature_names=None, stacked: bool = False):
    n_target, n_features = model.coef_.shape
    ix = feature_names if feature_names is not None else list(range(n_features))
    cols = (
        target_names[:n_target]
        if target_names is not None
        else ["class_{}".format(i) for i in range(n_target)]
    )
    df = pd.DataFrame(index=ix, columns=cols, data=model.coef_.T)
    df.index.name = "Features"
    df.columns.name = "output_class"
    bar = df.hvplot.bar(legend=True, stacked=stacked, rot=75)
    bar = bar.opts(ylabel="Aggregated coefficients", title="Feature importances")
    return bar


def plot_model_evaluation(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    target_names=None,
    feature_names=None,
    normalize: bool = False,
    resolution: int = 100,
    stacked: bool = False,
):
    y_pred_test = model.predict(X_test)
    X, y = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])
    y_pred_total = model.predict(X)
    metrics = plot_classification_report(y=y_test, y_pred=y_pred_test, target_names=target_names)
    conf_mat = plot_confussion_matrix(
        y_test=y_test, y_pred=y_pred_test, target_names=target_names, normalize=normalize
    )
    bounds = plot_decision_boundaries(X=X, y=y, y_pred=y_pred_total, resolution=resolution)
    features = plot_feature_importances(
        model=model, target_names=target_names, feature_names=feature_names, stacked=stacked
    )
    gspec = pn.GridSpec(
        min_height=1000, height=1000, max_height=1200, min_width=800, max_width=1980, width=800
    )
    gspec[1:3, 0:2] = bounds
    gspec[0, 2:4] = metrics
    gspec[0, 0:2] = pn.pane.HTML(str(model), margin=0)
    gspec[1:3, 2:4] = conf_mat
    gspec[3:5, :] = features
    return gspec


def interactive_logistic_regression(
    X, y, target_names=None, feature_names=None, stacked: bool = False
):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    def interactive_model(C, penalty, fit_intercept, intercept_scaling, l1_ratio, class_weight):
        model = LogisticRegression(C=C, penalty=penalty, solver="saga",
                                   fit_intercept=fit_intercept,
                                   intercept_scaling=intercept_scaling,
                                   l1_ratio=l1_ratio, class_weight=class_weight,
                                   random_state=42).fit(X, y)
        return plot_model_evaluation(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            target_names=target_names,
            feature_names=feature_names,
            stacked=stacked,
        )

    c_slider = widgets.LiteralInput(value=1, name="C")
    penalty_tog = widgets.RadioButtonGroup(
        name="penalty", options=["none", "l1", "l2", "elasticnet"]
    )
    fit_intercept = widgets.Toggle(name="fit_intercept")
    intercept_scaling = widgets.LiteralInput(value=1, name="intercept_scaling")
    l1_ratio = widgets.FloatSlider(end=1., start=0., value=0.5, name="l1_ratio")
    class_weight = widgets.LiteralInput(value=None, name="class_weight")
    return pn.interact(interactive_model, C=c_slider, penalty=penalty_tog,
                       fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                       l1_ratio=l1_ratio, class_weight=class_weight)


def plot_dataset_2d(X_train, y_train, X_test, y_test):

    idx = y_train == 0
    idx_test = y_test == 0
    clase_0_train = hv.Scatter(
        {"x": X_train[idx, 0], "y": X_train[idx, 1]}, label="Clase 0 train"
    ).opts(color="b", size=20, alpha=0.5, tools=["hover"])
    clase_1_train = hv.Scatter(
        {"x": X_train[~idx, 0], "y": X_train[~idx, 1]}, label="Clase 1 train"
    ).opts(color="r", size=20, alpha=0.5, tools=["hover"])
    clase_0_test = hv.Scatter(
        {"x": X_test[idx_test, 0], "y": X_test[idx_test, 1]}, label="Clase 0 test"
    ).opts(color="b", size=20, alpha=0.5, marker="s", tools=["hover"])
    clase_1_test = hv.Scatter(
        {"x": X_test[~idx_test, 0], "y": X_test[~idx_test, 1]}, label="Clase 1 test"
    ).opts(color="r", size=20, alpha=0.5, marker="s", tools=["hover"])
    plot = clase_0_train * clase_1_train * clase_0_test * clase_1_test
    plot = plot.opts(
        width=550,
        height=450,
        title="Dataset de ejemplo",
        xlabel="Feature 1",
        ylabel="Feature 2",
        legend_position="top",
        xlim=(7.5, 12.3),
        ylim=(-1.5, 6),
    )
    return plot


class RegLog:
    """
    Modelo de regresión logistica.

    Este modelo solo puede usarse para clasificación binaria (clases {1, 0}). \
     Una vez entrenado, modela la probabilidad de que un ejemplo pertenezca a la clase 1.

     Puede ser usado para predecir la probabilidad de pertenecer a la clase uno, o para
     predecir directamente la clase asignada a cada ejemplo.
    """

    def __init__(self, learning_rate: float = 0.001, num_iters: int = 50_000):
        """
        Inicializa una instancia de Reglog.

        Args:
            learning_rate: Este valor define cuanto va a actualizarse el model con su \
                           correspondiente gradiente en cada iteración de entrenamiento.
            num_iters: Número de iteraciones  que se van a llevar a cabo para \
                        entrenar el modelo.
        """
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.weights = None  # Será un vector de longitud igual al numero de features + 1.

    def __repr__(self) -> str:
        string = "{}\n".format(self.__class__.__name__)
        if self.weights is None:
            string += "Modelo no entrenado."
        else:
            w_columns = ["weight_{}: {:.4f} ".format(i, w) for i, w in enumerate(self.weights[1:])]
            string += "".join(w_columns)
            string += "intercept: {:.4f}".format(self.weights[0])
        return string

    @staticmethod
    def sigmoid(linear_output: np.ndarray) -> np.ndarray:
        """
        Aplica la función sigmoide a cada uno de los valores del array de entrada.

        Args:
            linear_output: Corresponde a la predicción de un modelo lineal.

        Returns:
            Aray que contiene el resultado de aplicar la función sigmoide al \
            valor de entrada.
        """
        return 1 / (1 + np.exp(-linear_output))

    @staticmethod
    def log_likelihood(sigmoid_output, y: np.ndarray) -> np.ndarray:
        """
        Función de log-likehood que puede usarse para cuantificar el error del modelo.

        Args:
            sigmoid_output: Predicción del modelo en forma de probabilidad.
            y: Valor real correspondiente a cada predicción realizada por el modelo.

        Returns:
            Logaritmo de la función de verosimilitud correspondiente a los valores de entrada.
        """
        return (-y * np.log(sigmoid_output) - (1 - y) * np.log(1 - sigmoid_output)).mean()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Para cada ejemplo, predice la probabilidad de pertenecer a la clase 1.

        Args:
            X: Dataset que se desea clasificar. Cada ejemplo corresponde a una \
               fila (dimensión 0 del array), y cada feature a una columna \
               (dimensión 1 del array).

        Returns:
            Array que contiene las probabilidades assignadas a cada ejemplo de pertenecer a la
            clase 1. Este vector contiene floats en el intervalo [0, 1].
        """
        # El efecto del intercept es sumar una constante a la predicción lineal del modelo.
        # Al aplicar el producto escalar el peso asignado al intercept se multiplica por 1,
        # lo cual equivale a sumar el peso, pero es más eficiente de calcular.
        intercept = np.ones((X.shape[0], 1))
        x = np.concatenate((intercept, X), axis=1)
        return self.sigmoid(np.dot(x, self.weights))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la classe a la que pertenecen cada uno de los ejemplos de X.

        Args:
            X: Dataset que se desea clasificar. Cada ejemplo corresponde a una \
               fila (dimensión 0 del array), y cada feature a una columna \
               (dimensión 1 del array).

        Returns:
            Devuelve un array unidmiensional que contiene las predicciones para cada ejemplo de X.
            Este vector solo puede contener los valores {0, 1}
        """
        return self.predict_prob(X).round()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entrena el modelo de regresión logística.

        Args:
            X_train: Dataset de entrenamiento. Este array contiene \
               las features como columnas y los ejemplos de \
               entrenamiento como filas.

            y_train: Classes a las que pertenecen los ejemplos \
               del dataset de entrenamiento codificadas como \
               1 o 0. Cada elemento de este vector corresponde \
               al ejemplo x con el mismo índice. (Un elemento por \
               cada fila de x)
        """
        self._initialize_weights(X_train)
        for i in range(self.num_iters):
            self.training_iteration(X_train, y_train)

    def training_iteration(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Actualiza los pesos del modelo realizando una iteración del algoritmo gradient descend.
        """
        prediction = self.predict_proba(X_train)
        loss = self.log_likelihood(prediction, y_train)
        # Guardar estos valores es un truco para calcular el gradiente mas eficientemente.
        self._prediction, self._y_train, self._X_train = prediction, y_train, X_train
        # En general, el gradiente se calcula derivando la función
        # de pérdida usando la regla de la cadena.
        loglike_gradient = self._calculate_gradient(loss)
        self.weights -= self.learning_rate * loglike_gradient

    def _initialize_weights(self, X: np.ndarray) -> None:
        """Inicializa los pesos del modelo."""
        n_features = X.shape[1]
        self.weights = np.zeros(n_features + 1)  # El + 1 corresponde al peso de intercept.

    def _calculate_gradient(self, loss: np.ndarray) -> np.ndarray:
        """
        Devuelve el gradiente de la función log likelihood.

        Args:
            loss: Función de coste que va a derivarse. Será ignorado porque utilizaremos un truco \
                  matemático que no requiere del valor de la función de loss, y que es mas
                  eficiente de calcular.

        Returns:
            Array que contiene el gradiente de loss.
        """
        # Con un poco de matemáticas, podemos darnos cuenta de que el gradiente del
        # log-likelihood no es más que la multiplicación de la matriz de entradas transpuesta (X.T)
        # por el error de la predicción.
        intercept = np.ones((self._X_train.shape[0], 1))
        X = np.concatenate((intercept, self._X_train), axis=1)
        gradient = np.dot(X.T, (self._prediction - self._y_train)) / self._y_train.size
        return gradient


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


def plot_classes(model, X, y):
    try:
        probs = model.predict_proba(X)[:, 1]
    except:
        probs = model.predict_proba(X)
    data = pd.DataFrame(
        {"x": X[:, 0], "y": X[:, 1], "target": y, "prob": probs, "predicted": probs.round()}
    )
    return hv.Scatter(data).opts(size=10, color="target", tools=["hover"], cmap=["blue", "red"])


def plot_boundary(model, min_x, max_x):
    theta = np.concatenate(
        [model.intercept_, model.coef_[0]]
    )  # getting the x co-ordinates of the decision boundary
    plot_x = np.array([min_x, max_x])
    # getting corresponding y co-ordinates of the decision boundary
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])  # Plotting the Single Line Decision
    # Boundary
    data = pd.DataFrame({"x": plot_x, "y": plot_y})
    return hv.Curve(data).opts(color="#cfcb02")


def plot_probability_grid(model, X):
    probabilities, xs, ys = predict_grid(model, X)
    data = pd.DataFrame()
    qmesh = hv.QuadMesh((xs, ys, probabilities)).opts(
        colorbar=True, width=500, height=400, cmap="YlGnBu"
    )
    return qmesh


def plot_model_output(model, X, y):

    # img = plot_interactive_image(probabilities)
    points = plot_classes(model, X, y)
    boundary = plot_boundary(
        model, safe_margin(X[:, 0].min(), True, 0.075), safe_margin(X[:, 0].max(), False, 0.075)
    )
    boundary = boundary.opts(line_width=5)
    qmesh = plot_probability_grid(model, X)
    return qmesh * points * boundary


class RegLogTrainingPlotter:
    loss_plot_columns = ["train_loss", "test_loss"]

    def __init__(self, reglog: RegLog, max_weights: int = 10, plot_every: int = 10):
        self.reglog = reglog
        self.max_weights = max_weights
        self.plot_every = plot_every

        self._loss_df = pd.DataFrame(
            {col: [] for col in self.loss_plot_columns}, columns=self.loss_plot_columns
        )
        self._loss_register = StreamzDataFrame(example=self._loss_df)

        self._df_weights = pd.DataFrame(
            {"w_{}".format(i): [] for i in range(max_weights)},
            columns=["w_{}".format(i) for i in range(max_weights)],
        )
        self._weights_register = StreamzDataFrame(example=self._df_weights)
        self.curr_iter = 0
        self.train_loss = None
        self.test_loss = None
        self._loss_plot = None
        self._weights_plot = None

    def __repr__(self):
        return self.reglog.__repr__()

    def fit(self, X, y, X_test, y_test):
        self.create_plot()
        self.reglog._initialize_weights(X)
        for i in range(self.reglog.num_iters):
            self.curr_iter = i
            self.reglog.training_iteration(X, y)
            prediction = self.reglog.predict_proba(X)
            self.train_loss = self.reglog.log_likelihood(prediction, y)
            test_prediction = self.reglog.predict_proba(X_test)
            self.test_loss = self.reglog.log_likelihood(test_prediction, y_test)
            self._record_loss()
            self._record_weights()

    def _record_loss(self):
        if self.curr_iter % self.plot_every == 0:
            loss_df = pd.DataFrame(
                {"train_loss": [self.train_loss], "test_loss": [self.test_loss]},
                columns=["train_loss", "test_loss"],
                index=[self.curr_iter],
            )
            self._loss_register.emit(loss_df)

    def _record_weights(self):
        if self.curr_iter % self.plot_every == 0:
            n_weights = self.max_weights
            df_w = pd.DataFrame(
                {
                    "w_{}".format(ix_w): [w]
                    for ix_w, w in enumerate(self.reglog.weights[:n_weights])
                },
                columns=["w_{}".format(ix) for ix in range(n_weights)],
                index=[self.curr_iter],
            )
            self._weights_register.emit(df_w)

    def create_plot(self):
        title = " {} Evolution of the model during training".format(self.reglog.__class__.__name__)
        self._loss_plot = self._create_loss_plot()
        self._weights_plot = self._create_weights_plot()
        return (self._loss_plot + self._weights_plot).opts(title=title)

    def _create_loss_plot(self):
        n_iters = self.reglog.num_iters
        losses = self._loss_register.hvplot.line(ylim=(0, None))
        losses = losses.opts(
            ylim=(0, None), xlim=(0, n_iters + int(n_iters / 20)), title="Loss", width=400
        )
        return losses

    def _create_weights_plot(self):
        bars = self._weights_register.hvplot.bar(stacked=True, rot=75, shared_axes=False)
        bars = bars.opts(
            width=400, title="Model Weights", xlabel="Training iteration", ylabel="Weight value"
        )
        return bars


class ModelPlotter:
    @staticmethod
    def plot_interactive_image(grid):
        img = hv.Image(grid)
        # Declare pointer stream initializing at (0, 0) and linking to Image
        pointer = streams.PointerXY(x=0, y=0, source=img)
        # Define function to draw cross-hair and report value of image at location as text

        def cross_hair_info(x, y):
            text = hv.Text(
                x + 0.05, y, "%.3f %.3f %.3f" % (x, y, img[x, y]), halign="left", valign="bottom"
            )
            return hv.HLine(y) * hv.VLine(x) * text

        # Overlay image and cross_hair_info
        return img * hv.DynamicMap(cross_hair_info, streams=[pointer])

    @staticmethod
    def plot_dataset(X, y, model=None, dim_0: int = 0, dim_1: int = 1, cmap: str = "viridis"):
        data = pd.DataFrame({"x": X[:, dim_0], "y": X[:, dim_1], "target": y})
        if model:
            probs = model.predict_proba(X)
            classes = model.predict(X)
            data["prediction"] = classes
            for i in range(probs.shape[1]):
                data["prob_class_{}".format(i)] = probs[:, i]
        return hv.Scatter(data).opts(size=10, color="target", tools=["hover"], cmap=cmap)

    @classmethod
    def plot_decision_boundaries(
        cls, model, X, class_names: str = None, dim_x: int = 0, dim_y: int = 1
    ):
        """Plot the decission boundaries of a classification model."""

        min_x = cls.safe_bound(X[:, dim_x].min(), low=True, pct=0.075)
        max_x = cls.safe_bound(X[:, dim_y].max(), low=False, pct=0.075)
        plots = {}
        for i, (intercept, coef) in enumerate(
            zip(model.intercept_.tolist(), model.coef_.tolist())
        ):
            # getting the x co-ordinates of the decision boundary
            theta = np.concatenate([[intercept], coef])
            plot_x = np.array([min_x, max_x])
            # getting corresponding y co-ordinates of the decision boundary
            plot_y = (-1 / coef[dim_y]) * (
                theta[dim_x] * plot_x + intercept
            )  # Plotting the Single Line Decision
            # Boundary
            data = pd.DataFrame({"x": plot_x, "y": plot_y})
            boundary = hv.Curve(data).opts(line_width=5)
            name = "class_{}".format(i) if class_names is None else class_names[i]
            plots[name] = boundary
        return hv.NdOverlay(plots, kdims="Decision boundaries")

    @classmethod
    def plot_probability_grid(cls, model, X):
        probabilities, xs, ys = cls.predict_grid(model, X)
        qmesh = hv.QuadMesh((xs, ys, probabilities)).opts(tools=["hover"])
        return qmesh

    @classmethod
    def plot_binary_clasification(cls, model, X, y):
        points = cls.plot_dataset(X, y, model=model)
        boundary = cls.plot_decision_boundaries(model, X)
        qmesh = cls.plot_probability_grid(model, X)
        return qmesh * points * boundary

    @classmethod
    def plot_classification(cls, model, X, y, dim_x: int = 0, dim_y: int = 1):
        points = cls.plot_dataset(X, y, model=model)
        boundary = cls.plot_decision_boundaries(model, X, dim_x=dim_x, dim_y=dim_y)
        qmesh = cls.plot_regions(model, X)
        return qmesh * points * boundary

    @classmethod
    def plot_regions(cls, model, X):
        regions, xs, ys = cls.predict_classes(model, X)
        qmesh = hv.QuadMesh((xs, ys, regions)).opts(tools=["hover"])
        return qmesh

    @staticmethod
    def safe_bound(val, low=True, pct: float = 0.05):
        low_pct, high_pct = 1 - pct, 1 + pct
        func = min if low else max
        return func(val * low_pct, val * high_pct)

    @classmethod
    def safe_bounds_fom_examples(cls, X, x_col: int = 0, y_col: int = 1):
        low_x, high_x = X[:, x_col].min(), X[:, x_col].max()
        low_x = cls.safe_bound(low_x)
        high_x = cls.safe_bound(high_x, False)
        low_y, high_y = X[:, y_col].min(), X[:, y_col].max()
        low_y = cls.safe_bound(low_y)
        high_y = cls.safe_bound(high_y, False)
        return low_x, high_x, low_y, high_y

    @classmethod
    def example_meshgrid(cls, X, n_bins=100, low_th: float = 0.95, high_th: float = 1.05):
        low_x, high_x, low_y, high_y = cls.safe_bounds_fom_examples(X)
        xs = np.linspace(low_x, high_x, n_bins)
        ys = np.linspace(low_y, high_y, n_bins)
        return np.meshgrid(xs, ys)

    @classmethod
    def predict_grid(cls, model, X):
        if X.shape[1] != 2:
            raise ValueError("Predict grid only accepts datasets with 2 features")
        x_grid, y_grid = cls.example_meshgrid(X)
        grid = np.c_[x_grid.ravel(), y_grid.ravel()]
        probs = model.predict_proba(grid)[:, 1].reshape(x_grid.shape)
        return probs, x_grid, y_grid

    @classmethod
    def predict_classes(cls, model, X):
        x_grid, y_grid = cls.example_meshgrid(X)
        grid = np.c_[x_grid.ravel(), y_grid.ravel()]
        probs = model.predict(grid).reshape(x_grid.shape)
        return probs, x_grid, y_grid
