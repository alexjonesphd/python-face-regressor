# python-face-regressor

``pythonfaceregressor`` is a Python package for working with face images that goes beyond facial averaging, distributed under the 3-Clause BSD License.

Given a set of facial photographs, associated facial landmarks, and a set of attributes for those faces (e.g., perceived attractiveness or trustworthiness), the package learns the relationships between pixel and coordinate elements and attributes using regression. Users can then use the learned model to predict facial appearances for one attribute while controlling entirely for another, or specify combinations of predictors that would not be possible using facial averaging. The model also provides a set of attributes for analysis, such as pixel-by-pixel relationships with facial attributes, standard error maps, image warping, and a powerful visualiser tool.


## Installation

### Dependencies

``pythonfaceregressor`` requires:
