import numpy as np
import pandas as pd

import holoviews as hv
import hvplot.streamz
import hvplot
import hvplot.pandas
from holoviews import streams

import streamz
from streamz.dataframe import DataFrame as StreamzDataFrame


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

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
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
        prediction = self.predict_prob(X_train)
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

def safe_margin(val, low=True, pct: float=0.05):
    low_pct, high_pct = 1 - pct, 1 + pct
    func = min if low else max
    return func(val * low_pct, val * high_pct)
    

def example_meshgrid(X, n_bins=10, low_th: float=0.95, high_th:float=1.05):
    low_x, high_x = X[:, 0].min(), X[:, 0].max()
    low_x = safe_margin(low_x)
    high_x = safe_margin(high_x, False)
    low_y, high_y = X[:, 1].min(), X[:, 1].max()
    low_y = safe_margin(low_y)
    high_y = safe_margin(high_y, False)
    xs = np.linspace(low_x, high_x, n_bins)
    ys = np.linspace(low_y, high_y, n_bins)
    return np.meshgrid(xs, ys)

def predict_grid(model, X):
    x_grid, y_grid = example_meshgrid(X)
    grid = np.c_[x_grid.ravel(), y_grid.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(x_grid.shape)
    return probs, x_grid, y_grid

def plot_interactive_image(grid):
    img = hv.Image(grid)
    # Declare pointer stream initializing at (0, 0) and linking to Image
    pointer = streams.PointerXY(x=0, y=0, source=img)
    # Define function to draw cross-hair and report value of image at location as text
    def cross_hair_info(x, y):
        text = hv.Text(x+0.05, y, '%.3f %.3f %.3f'% (x, y, img[x,y]), halign='left', valign='bottom')
        return hv.HLine(y) * hv.VLine(x) * text
    # Overlay image and cross_hair_info
    return img * hv.DynamicMap(cross_hair_info, streams=[pointer])

def plot_classes(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    data = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "target": y, "porb": probs})
    return hv.Scatter(data).opts(size=10, color="target", tools=["hover"], cmap="viridis")

def plot_boundary(model, min_x, max_x):
    theta = np.concatenate([model.intercept_, model.coef_[0]])# getting the x co-ordinates of the decision boundary
    plot_x = np.array([min_x, max_x])
    # getting corresponding y co-ordinates of the decision boundary
    plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0]) # Plotting the Single Line Decision
    # Boundary
    data = pd.DataFrame({"x": plot_x, "y": plot_y})
    return hv.Curve(data)

def plot_probability_grid(model, X):
    probabilities, xs, ys = predict_grid(model, X)
    data = pd.DataFrame()
    qmesh = hv.QuadMesh((xs, ys, probabilities))
    return qmesh

def plot_model_output(model, X, y):
    
    #img = plot_interactive_image(probabilities)
    points = plot_classes(model, X, y)
    boundary = plot_boundary(model, safe_margin(X[:, 0].min(), True, 0.075),
                             safe_margin(X[:,0].max(), False, 0.075))
    boundary = boundary.opts(line_width=5)
    qmesh = plot_probability_grid(model, X)
    return qmesh * points * boundary


class RegLogTrainingPlotter:
    loss_plot_columns = ['train_loss', 'test_loss']

    def __init__(self, reglog: RegLog, max_weights: int = 10, plot_every: int = 10):
        self.reglog = reglog
        self.max_weights = max_weights
        self.plot_every = plot_every

        self._loss_df = pd.DataFrame({col: [] for col in self.loss_plot_columns},
                                     columns=self.loss_plot_columns)
        self._loss_register = StreamzDataFrame(example=self._loss_df)

        self._df_weights = pd.DataFrame({"w_{}".format(i): [] for i in range(max_weights)},
                                        columns=["w_{}".format(i) for i in range(max_weights)])
        self._weights_register = StreamzDataFrame(example=self._df_weights)
        self.curr_iter = 0
        self.train_loss = None
        self.test_loss = None
        self._loss_plot = None
        self._weights_plot = None

    def __repr__(self):
        return self.reglog.__repr__()

    def fit(self, X, y, X_test, y_test):
        self.reglog._initialize_weights(X)
        for i in range(self.reglog.num_iters):
            self.curr_iter = i
            self.reglog.training_iteration(X, y)
            prediction = self.reglog.predict_prob(X)
            self.train_loss = self.reglog.log_likelihood(prediction, y)
            test_prediction = self.reglog.predict_prob(X_test)
            self.test_loss = self.reglog.log_likelihood(test_prediction, y_test)
            self._record_loss()
            self._record_weights()

    def _record_loss(self):
        if self.curr_iter % self.plot_every == 0:
            loss_df = pd.DataFrame(
                {'train_loss': [self.train_loss], 'test_loss': [self.test_loss]},
                columns=['train_loss', 'test_loss'], index=[self.curr_iter])
            self._loss_register.emit(loss_df)

    def _record_weights(self):
        if self.curr_iter % self.plot_every == 0:
            n_weights = self.max_weights
            df_w = pd.DataFrame({"w_{}".format(ix_w): [w] for ix_w, w in
                                 enumerate(self.reglog.weights[:n_weights])},
                                columns=["w_{}".format(ix) for ix in range(n_weights)],
                                index=[self.curr_iter])
            self._weights_register.emit(df_w)

    def create_plot(self):
        title = " {} Evolution of the model during training".format(self.reglog.__class__.__name__)
        self._loss_plot = self._create_loss_plot()
        self._weights_plot = self._create_weights_plot()
        return (self._loss_plot + self._weights_plot).opts(title=title)

    def _create_loss_plot(self):
        n_iters = self.reglog.num_iters
        losses = self._loss_register.hvplot.line(ylim=(0, None))
        losses = losses.opts(ylim=(0, None), xlim=(0, n_iters + int(n_iters / 20)),
                             title="Loss", width=400)
        return losses

    def _create_weights_plot(self):
        bars = self._weights_register.hvplot.bar(stacked=True,
                                                 rot=75, shared_axes=False)
        bars = bars.opts(width=400, title="Model Weights", xlabel='Training iteration',
                         ylabel='Weight value')
        return bars


class ModelPlotter:

    def plot_interactive_image(self, grid):
        img = hv.Image(grid)
        # Declare pointer stream initializing at (0, 0) and linking to Image
        pointer = streams.PointerXY(x=0, y=0, source=img)

        # Define function to draw cross-hair and report value of image at location as text
        def cross_hair_info(x, y):
            text = hv.Text(x + 0.05, y, '%.3f %.3f %.3f' % (x, y, img[x, y]), halign='left',
                           valign='bottom')
            return hv.HLine(y) * hv.VLine(x) * text

        # Overlay image and cross_hair_info
        return img * hv.DynamicMap(cross_hair_info, streams=[pointer])

    def plot_classes(self, model, X, y):
        probs = model.predict_proba(X)[:, 1]
        data = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "target": y, "porb": probs})
        return hv.Scatter(data).opts(size=10, color="target", tools=["hover"], cmap="viridis")

    def plot_boundary(self, model, min_x, max_x):
        theta = np.concatenate([model.intercept_, model.coef_[
            0]])  # getting the x co-ordinates of the decision boundary
        plot_x = np.array([min_x, max_x])
        # getting corresponding y co-ordinates of the decision boundary
        plot_y = (-1 / theta[2]) * (
                    theta[1] * plot_x + theta[0])  # Plotting the Single Line Decision Boundary
        data = pd.DataFrame({"x": plot_x, "y": plot_y})
        return hv.Curve(data)

    def plot_probability_grid(self, model, X):
        probabilities, xs, ys = self.predict_grid(model, X)
        data = pd.DataFrame()
        qmesh = hv.QuadMesh((xs, ys, probabilities)).opts(tools=["hover"])
        return qmesh

    def plot_model_output(self, model, X, y):
        # img = plot_interactive_image(probabilities)
        points = self.plot_classes(model, X, y)
        boundary = self.plot_boundary(model, self.safe_bound(X[:, 0].min(), True, 0.075),
                                      self.safe_bound(X[:, 0].max(), False, 0.075))
        boundary = boundary.opts(line_width=5)
        qmesh = self.plot_probability_grid(model, X)
        return qmesh * points * boundary

    @staticmethod
    def safe_bound(val, low=True, pct: float = 0.05):
        low_pct, high_pct = 1 - pct, 1 + pct
        func = min if low else max
        return func(val * low_pct, val * high_pct)

    def safe_bounds_fom_examples(self, X, x_col: int=0, y_col: int=1 ):
        low_x, high_x = X[:, x_col].min(), X[:, x_col].max()
        low_x = self.safe_bound(low_x)
        high_x = self.safe_bound(high_x, False)
        low_y, high_y = X[:, y_col].min(), X[:, y_col].max()
        low_y = self.safe_bound(low_y)
        high_y = self.safe_bound(high_y, False)
        return low_x, high_x, low_y, high_y

    def example_meshgrid(self, X, n_bins=10, low_th: float = 0.95, high_th: float = 1.05):
        low_x, high_x, low_y, high_y = self.safe_bounds_fom_examples(X)
        xs = np.linspace(low_x, high_x, n_bins)
        ys = np.linspace(low_y, high_y, n_bins)
        return np.meshgrid(xs, ys)

    def predict_grid(self, model, X):
        x_grid, y_grid = self.example_meshgrid(X)
        grid = np.c_[x_grid.ravel(), y_grid.ravel()]
        probs = model.predict_proba(grid)[:, 1].reshape(x_grid.shape)
        return probs, x_grid, y_grid


    
