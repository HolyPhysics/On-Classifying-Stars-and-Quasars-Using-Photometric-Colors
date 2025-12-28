# On Classifying Stars and Quasars Using Photometric Colors

Quasars are galaxies whose central supermassive black holes are accreting matter at such extreme rates that they outshine their host galaxies. Because both quasars and stars appear as unresolved point sources in imaging surveys, distinguishing between them using photometry alone is a non-trivial classification problem even though photometry shows there exists some distintion between the duo.

In this project, I apply seven different machine learning models to classify quasars and stars from data derived from the Sloan Digital Sky Survey(SDSS). Using metrics including but not limited to ROC curve, confusion matrix, accuracy, precision, and recall to explore each models effectiveness in tackling the problem.

<!-- This project applies machine-learning classification techniques to photometric color indices derived from Sloan Digital Sky Survey (SDSS) data in order to separate quasars from stars. Multiple classifiers are evaluated and compared using ROC curves, confusion matrices, and classification reports. -->

---

<!-- ## Scientific Motivation

Modern astronomical surveys generate catalogs containing millions of unresolved point sources. While spectroscopy provides reliable object classification, it is observationally expensive. Photometric classification of quasars enables efficient spectroscopic target selection, large-scale structure studies, and investigations of quasar evolution and the early universe.

--- -->

## Methods and Data

### Data Sources
- Stars from the SDSS Stellar Parameter Pipeline (SSPP), accessed via astroML
- Quasars from the SDSS DR7 quasar catalog

### Features
Photometric color indices constructed from SDSS ugriz magnitudes:
- u − g, g - r, r - i, i - z 

### Labels
Binary classification:
- 0: Star, 1: Quasar
<!--  -->

### Class Balancing
Equal-sized samples are drawn from each class to prevent bias due to class imbalance.

### Models Used
- Gaussian Naive Bayes
- Gaussian Mixture Model (Bayesian)
- k-Nearest Neighbors(KNN)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Decision Tree (with optimized max_depth)
- Random Forest (with optimized max_depth)

### Hyperparameter Optimization
During each trainin session, cross-validation was performed on the KNN model to determine the optimal value for the _*n-neighbors*_ hyperparameter. For the tree-based model, GridSearchCV was used to performed cross-validation on the models and the _*n_depth*_ was set to the recovered _*best_depth*_ value. 

---

## Installation

### Requirements

- Python 3.x
- numpy
- matplotlib
- astropy
- astroML
- scikit-learn

### Installation Command

```bash
pip install numpy matplotlib astropy astroML scikit-learn
```

---
<!-- 
## How to Run

### Step 1: Preprocessing
 -->
