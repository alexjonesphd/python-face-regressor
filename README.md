# python-face-regressor

``pythonfaceregressor`` is a Python package for working with face images that goes beyond facial averaging, distributed under the 3-Clause BSD License.

Given a set of facial photographs, associated facial landmarks, and a set of attributes for those faces (e.g., perceived attractiveness or trustworthiness), the package learns the relationships between pixel and coordinate elements and attributes using regression. Users can then use the learned model to predict facial appearances for one attribute while controlling entirely for another, or specify combinations of predictors that would not be possible using facial averaging. The model also provides a set of attributes for analysis, such as pixel-by-pixel relationships with facial attributes, standard error maps, image warping, and a powerful visualiser tool.


## Installation

### Dependencies

``pythonfaceregressor`` requires the following packages, and full functionality requires a Jupyter notebook.

* Python (>= 3.6)
* NumPy (>= 1.14.0)
* SciPy (>= 1.0.0)
* pandas (>= 0.22.0)
* scikit-image (>= 0.13.1)
* Python Image Library (>= 5.0.0)
* OpenCV2 (>= 3.3.1)

Running the ``visualiser`` module of ``pythonfaceregressor`` also requires Bokeh >= 0.12.13, running from within a Jupyter notebook.

### User installation
The easiest way to obtain the full range of dependecies for the package is to install the [Anaconda distribution](https://www.anaconda.com/download/), which provides stable releases of all the data science tools depends on, except for OpenCV. To install OpenCV, run from the command line:

``pip install opencv-python``

This should install the stable Python 3.4 version of OpenCV, that works well with the Anaconda distribution.

Install the ``pythonfaceregressor`` package by runnig the following from the command line:

``pip install pythonfaceregressor``

For full functionality, open a Jupyer notebook and try:

``import pythonfaceregressor as pyfacer``

### User instructions
Full instructions on using the package can be found in the accompanying Jupyter notebook, and in the academic publication below.

### Citations
If you use the package in a scientific publication, please cite the following forthcoming paper:

INSERT PAPER TITLE HERE

