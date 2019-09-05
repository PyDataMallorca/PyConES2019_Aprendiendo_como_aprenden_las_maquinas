# PyConES2019: Aprendiendo cómo aprenden las máquinas

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/PyDataMallorca/PyConES2019_Aprendiendo_como_aprenden_las_maquinas/master)

Taller "Aprendiendo cómo aprenden las máquinas" preparado por los organizadores de [PyDataMallorca](https://twitter.com/PyDataMallorca) para la [PyConES2019](https://2019.es.pycon.org/).

# Preparación imprescindible antes del taller

**NOTA: es muy importante que acudas al taller con todo instalado, para que puedas aprovechar al máximo todo lo que explicaremos.**

Necesitarás lo siguiente:

* Un portátil.
* Descargar los materiales para el curso y descomprimirlos en el portátil ([usa este enlace para descargarlos](https://github.com/PyDataMallorca/FTW2019_Introduccion_a_data_science_en_Python/archive/master.zip)).
* Python 3.6 o superior instalado.
* Las librerías que vamos a usar (Jupyter, Numpy, Matplotlib, Pandas y Scikit-Learn).

Formas de conseguir lo anterior:

**Instalación de Anaconda**

La forma más sencilla sería instalando Anaconda para vuestro sistema operativo. La distribución Anaconda junto con instrucciones de cómo instalarlo lo podéis encontrar [en este enlace](https://www.anaconda.com/download/) (seleccionad la versión que incluya Python 3.6 o superior). La distribución Anaconda os instalará Python y un montón de paquetes que se usan en muchos ámbitos de Data Science y de la ciencia e ingeniería en general. Os dejamos también vídeos que hemos preparado para facilitaros el proceso:

* Para instalar Anaconda en Linux: https://www.youtube.com/watch?v=b9LV1J7vPuw&t=192s
* Para instalar Anaconda en MacOS: seguid las instrucciones de Linux de la línea anterior.
* Para instalar Anaconda en Windows: https://www.youtube.com/watch?v=MSnNTODnSBg

**Instalación de Miniconda**

Una segunda forma sería instalando Miniconda. Lo podéis descargar [desde este enlace](https://conda.io/miniconda.html) (seleccionad la versión que incluya Python 3.6 o superior). Una vez instalado MiniConda tenéis Python y una serie de utilidades instaladas. Os dejamos también vídeos que hemos preparado para facilitaros el proceso:

* Para instalar Miniconda en linux: https://www.youtube.com/watch?v=liqnwft_cbs
* Para instalar Miniconda en MacOS: Seguid las instrucciones de Linux de la línea anterior.
* Para instalar Miniconda en windows: https://www.youtube.com/watch?v=aYhlDfGhwuU

**Instalación de paquetes específicos**

[PUEDES OMITIR ESTE PASO SI HAS INSTALADO ANACONDA. SI HAS INSTALADO MINICONDA CONTINUA LEYENDO]

Para instalar el resto de paquetes necesarios podéis abrir una terminal (Linux/Mac) o el AnacondaPrompt (Windows), ejecutad lo siguiente (dependiendo del sistema operativo en el que estéis deberéis ejecutar unas cosas u otras).

`cd ruta/a/la/carpeta/descargada/y/descomprimida` (Linux o Mac)

`cd C:\ruta\a\la\carpeta\descargada\y\descomprimida` (Windows)

`conda env create -f environment.yml` (Linux, Mac, Windows)

# Durante el taller

Como en el paso anterior habéis instalado los paquetes necesarios, solamente tenéis que activar el entorno creado y usar los paquetes. Para ello, en la misma terminal que los pasos anteriores deberéis ejecutar **[si tienes Anaconda instalado te puedes saltar el anterior paso]**:

`source activate pycones19`

Y finalmente también ejecutar jupyter notebook para acceder al tutorial. Atención: se abrirá un navegador web.

`jupyter notebook`

# Resumen de la propuesta

Taller de 3 horas destinado a comprender el funcionamiento de los algoritmos más importantes en Data Science, mediante una explicación muy simple diseñada para todos los públicos y con ejercicios prácticos en Python, impartido por el equipo organizador de PyData Mallorca. En este taller veremos cuándo debemos aplicar y cómo funcionan las regresiones lineales, las regresiones logísticas y los árboles de decisión sin entrar en detalles matemáticos ni estadísticos complejos, pero permitiendo que el alumnado desarrolle una intuición clara sobre cómo funcionan estos algoritmos internamente y así poderlos aplicar a otras situaciones después del taller con facilidad. Para la explicación y parte práctica, utilizaremos las siguientes herramientas y librerías:

* **Jupyter** para la edición de código Python y texto enriquecido, 
* **Pandas** para la carga y transformación de los datos que utilizaremos en los algoritmos, 
* **matplotlib** para la visualización de los datos y 
* **scikit-learn** para la ejecución, parametrización y comprobación de los algoritmos.

# Autores (por orden alfabético inverso)

* Antònia Tugores [twitter](https://twitter.com/antoniatugores).

* Juan Carlos González [twitter](https://twitter.com/jcgavella).

* Guillem Duran [twitter](https://twitter.com/Miau_DB).

* Kiko Correoso ([pybonacci.org](https://pybonacci.org), [twitter](https://twitter.com/Pybonacci)).

* Jordi Contestí.

# Contacto

Si deseas contactar con nosotros puedes hacerlo del siguiente modo:

* Envía un correo electrónico a: pydata-mallorca@googlegroups.com.
* Envía un tweet a: https://twitter.com/PyDataMallorca.
* Envía un mensaje vía meetup a: https://www.meetup.com/PyData-Mallorca.

![](./images/PyDataMallorca_logo.png)
