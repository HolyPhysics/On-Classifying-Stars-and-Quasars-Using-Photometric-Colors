import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from astroML.utils import completeness_contamination
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from astroML.classification import GMMBayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

file_path: str = "galaxyquasar.csv"

try:
    file_container: list[float, ...] = Table.read(file_path, format="ascii")
except Exception as except_one:
    print(f' {type(except_one).__name__} occured!')

    file_container: list[float, ...] = Table.read(file_path, format="csv")

# print(file_container.columns)
# print(file_container)

# data_to_visualize: pd.DataFrame = file_container.to_pandas()
# print(data_to_visualize)
# sns.pairplot(data_to_visualize)
# plt.show()

u_color, g_color, r_color, i_color, z_color = file_container["u"], file_container["g"], file_container["r"], file_container["i"], file_container["z"]
class_container = np.array(file_container["class"])

u_g_color_container = np.array(u_color - g_color)
g_r_color_container = np.array(g_color - r_color)
r_i_color_container = np.array(r_color - i_color)
i_z_color_container = np.array(i_color - z_color)

color_container_for_analysis = np.vstack([u_g_color_container,g_r_color_container, r_i_color_container, i_z_color_container]).T
scaler = StandardScaler()
scaler.fit_transform(color_container_for_analysis) # actualy this makes no difference here, so we can safely proceed without any worries about getting some very weird results
# print(color_containr_for_analysis)

class_integer_container_for_analysis = np.array(class_container == "QSO", dtype=int)
# print(class_container)
# print(class_integer_container_for_analysis)


X_train, X_test, y_train, y_test = train_test_split(color_container_for_analysis, class_integer_container_for_analysis, test_size = 0.25, random_state= 42 )
# print(X_train.shape)


def classifiers(data, labels, classifier_name) -> None:

    X_train, X_test = data[0], data[-1]
    y_train, y_test = labels[0], labels[-1]

    color_container = ["green", "red", "blue", "orange"]

    figure, ax_main = plt.subplots(figsize=(8.5, 7.5)) # should be left in place here for crucial line by line interpretation by python

    for i in range(4):

        if classifier_name == GMMBayes or classifier_name == KNeighborsClassifier:

            classifier_model = classifier_name(i+2)

        else: 
            classifier_model = classifier_name()


        classifier_model.fit(X_train[:, 0:i+1], y_train)

        # training prediction
        training_prediction = classifier_model.predict(X_train[:, 0:i+1])

        # validation predicton
        validation_prediction = classifier_model.predict(X_test[:, 0:i+1])
        
        y_prob = classifier_model.predict_proba(X_test[:, 0:i+1])[:, 1]  # this extracts the probability for tyhe Quasars!! Observed this from first printing out the results before writing this particular line of code!!

        false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_prob)
        # print(threshold

        ## do some lazy visualizations
        ax_main.plot(false_positive_rate, true_positive_rate, linestyle="--", color=color_container[i], label=f"{i+1}")
        ax_main.set_xlabel("False Positive Rate")
        ax_main.set_ylabel("True Positive Rate")
        ax_main.set_title(f"ROC for {classifier_name.__name__} Classifier")
        ax_main.legend(loc="best")

    figure.tight_layout()
    plt.show()

    # print(training_prediction)
    # print(validation_prediction)
    # print(y_prob.shape)
    # print(y_prob)
    # print(gaussian_naive_bayes_classifier.classes_)

    

if __name__ == "__main__":

    data = [X_train, X_test]
    labels = [y_train, y_test]

    classifiers(data, labels, GaussianNB)
    classifiers(data, labels, GMMBayes)
    classifiers(data, labels, KNeighborsClassifier)
    classifiers(data, labels, LDA)
    classifiers(data, labels, QDA)
    