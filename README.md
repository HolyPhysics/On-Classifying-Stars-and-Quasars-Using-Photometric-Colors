# Quasar–Star Classification Using Photometric Colors

Quasars are galaxies whose central supermassive black holes are accreting matter at such extreme rates that they outshine their host galaxies. Because both quasars and stars appear as unresolved point sources in imaging surveys, distinguishing between them using photometry alone is a non-trivial classification problem.

This project applies machine-learning classification techniques to photometric color indices derived from Sloan Digital Sky Survey (SDSS) data in order to separate quasars from stars. Multiple classifiers are evaluated and compared using ROC curves, confusion matrices, and classification reports.

---

## Scientific Motivation

Modern astronomical surveys generate catalogs containing millions of unresolved point sources. While spectroscopy provides reliable object classification, it is observationally expensive. Photometric classification of quasars enables efficient spectroscopic target selection, large-scale structure studies, and investigations of quasar evolution and the early universe.

---

## Methods and Data

### Data Sources
- Stars from the SDSS Stellar Parameter Pipeline (SSPP), accessed via astroML
- Quasars from the SDSS DR7 quasar catalog

### Features
Photometric color indices constructed from SDSS ugriz magnitudes:
- u − g
- g − r
- r − i
- i − z

### Labels
Binary classification:
- 0: star
- 1: quasar

### Class Balancing
Equal-sized samples are drawn from each class to prevent bias due to class imbalance.

### Models Used
- Gaussian Naive Bayes
- Gaussian Mixture Model (Bayesian)
- k-Nearest Neighbors
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Decision Tree (with optimized max_depth)
- Random Forest (with optimized max_depth)

### Hyperparameter Optimization
Tree-based models use GridSearchCV with cross-validation to determine the optimal max_depth parameter.

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

## How to Run

### Step 1: Preprocessing

