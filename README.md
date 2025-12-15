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

## Evaluation Metrics

Model performance is assessed using:
- Receiver Operating Characteristic (ROC) curves
- Confusion matrices
- Precision, recall, and F1-score

---

## Project Structure

```text
.
├── preprocessing.py
│   Loads raw SDSS data, constructs color indices,
│   balances classes, and prepares training/testing sets
├── color_color_plot.py
│   Generates color–color diagrams comparing stars and quasars
├── main_classification.py
│   Trains classifiers, performs hyperparameter tuning,
│   generates ROC curves, confusion matrices,
│   and classification reports
└── README.md

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

This project was developed and tested in a standard Python environment.
astroML and astropy are fully compatible with Conda-based environments.


How to Run
Step 1: Preprocessing

```bash
python preprocessing.py

This step loads stellar and quasar catalogs, balances class sizes, computes photometric color indices, and outputs shuffled feature matrices and labels.

Step 2: Color–Color Visualization

```bash
python color_color_plot.py

This produces scatter plots illustrating the separation between stars and quasars in color–color space.

Step 3: Classification and Evaluation

```bash
python main_classification.py


This step:
trains all classifiers,
performs hyperparameter tuning where applicable,
plots ROC curves,
displays confusion matrices,
prints classification reports.

Outputs
Running the full pipeline produces:
Color–color diagrams of stellar and quasar populations
ROC curves for all classifiers
Confusion matrices for optimized models
Classification reports including precision, recall, and F1-score
