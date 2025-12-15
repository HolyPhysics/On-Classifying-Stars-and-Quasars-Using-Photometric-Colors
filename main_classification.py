from preprocessing import processed_data

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from astroML.classification import GMMBayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


X, y, star_data, quasar_data = processed_data() # I can easily delete this and still have a fully functioning code, by the way, because I have already loaded the X and y.

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.65, random_state= 42, stratify=y)
# print(X_train.shape)


def classifiers_roc_curve(data, labels, classifier_name, criterion = "gini", max_depth = 12) -> None:

    X_train, X_test = data[0], data[-1]
    y_train, y_test = labels[0], labels[-1]

    color_container = ["green", "red", "blue", "orange"]
    
    learned_max_depth = []

    figure, ax_main = plt.subplots(figsize=(8.5, 7.5)) # should be left in place here for crucial line by line interpretation by python

    for i in range(4):

        if classifier_name == GMMBayes or classifier_name == KNeighborsClassifier:

            classifier_model = classifier_name(i+2)

        elif classifier_name ==  DecisionTreeClassifier or classifier_name ==  RandomForestClassifier:

            classifier_model = classifier_name(random_state=42)
            max_depth_container = np.arange(1, 21)

            grid_search = GridSearchCV(classifier_model, param_grid={"max_depth": max_depth_container}, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            max_depth = grid_search.best_params_["max_depth"]  # if you want to optimize the speed of your code here, make this a method in an object and store this max_depth iin a self.max_depth variable/container. That way, you'll avoid running GridSearchCv every single time !!!
            learned_max_depth.append(max_depth)

            if classifier_name ==  RandomForestClassifier:
              classifier_model = classifier_name(max_depth = max_depth, criterion=criterion, random_state=42, n_jobs = -1) # I want to split the work among all my computers processors using n_jobs = -1
            else:
              classifier_model = classifier_name(max_depth=max_depth, criterion=criterion, random_state=42)

            print(f' The {i+1}-th max_depth is {max_depth}')

        else:
            classifier_model = classifier_name()


        classifier_model.fit(X_train[:, 0:i+1], y_train)

        # training prediction
        # training_prediction = classifier_model.predict(X_train[:, 0:i+1]) No need for training predicitions here as well since we're not doing any cross validation

        # validation predicton
        # validation_prediction = classifier_model.predict(X_test[:, 0:i+1]) No, need for validation here! We could do that solely for the ensemble classifiers

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
    # figure.savefig(f"/content/drive/MyDrive/Colab Images/{classifier_name.__name__}_roc_curve.png") # I'll have to put this in my slide.
    plt.show()

    return learned_max_depth


def confusion_matrix_and_classification_report(model):
    classifier_model = model

    classifier_model.fit(X_train, y_train)

    y_pred = classifier_model.predict(X_test)
    # print(y_test.shape)
    # print(y_pred.shape)

    #now, we create the confusion matrix object
    confusion_matrix_container =  confusion_matrix(y_test, y_pred)
    class_names = ["star", "quasar"]
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_container, display_labels=class_names)
    confusion_matrix_display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    # plt.savefig(f"/content/drive/MyDrive/Colab Images/{model_name.__name__}_confusion_matrix.png") # I'll have to put this in my slide.
    plt.show()

    classification_report_container = classification_report(y_test, y_pred)

    return classification_report_container



if __name__ == "__main__":

    data = [X_train, X_test]
    labels = [y_train, y_test]

    # for GaussianNB
    classifiers_roc_curve(data, labels, GaussianNB) #makes roc curve
    classification_report_container = confusion_matrix_and_classification_report( GaussianNB() )
    print(f"Classification report for GaussianNB() is \n {classification_report_container}. " )

    # for GMMBayes
    classifiers_roc_curve(data, labels, GMMBayes) #makes roc curve
    classification_report_container = confusion_matrix_and_classification_report( GMMBayes(5) )
    print(f"Classification report for GMMBayes(5) is \n {classification_report_container}. " )

    # KNeighborsClassifier
    classifiers_roc_curve(data, labels, KNeighborsClassifier) #makes roc curve
    classification_report_container = confusion_matrix_and_classification_report( KNeighborsClassifier(5) )
    print(f"Classification report for KNeighborsClassifier(5) is \n {classification_report_container}. " )

    #  for LDA
    classifiers_roc_curve(data, labels, LDA) #makes roc curve
    classification_report_container = confusion_matrix_and_classification_report( LDA() )
    print(f"Classification report for LDA() is \n {classification_report_container}. " )

    # for QDA
    classifiers_roc_curve(data, labels, QDA) #makes roc curve
    classification_report_container = confusion_matrix_and_classification_report( QDA() )
    print(f"Classification report for QDA() is \n {classification_report_container}. " )

    # for DecisionTreeClassifier with criterion = "gini"
    learned_max_depth = classifiers_roc_curve(data, labels, DecisionTreeClassifier, "gini") #makes roc curve
    max_depth =  max( learned_max_depth ) # This code grabs the max_depth(which is a default_value) programmatically. The max_depth is the second default item in the function definition. Should be read from the printed depth from Gridserach CV printed above 

    classification_report_container = confusion_matrix_and_classification_report( DecisionTreeClassifier(max_depth=max_depth, criterion="gini", random_state=42) )
    print(f"Classification report for DecisionTreeClassifier(gini, max_depth={max_depth} ) is \n {classification_report_container}. " )

    # for RandomForestClassifier with criterion = "entropy"
    learned_max_depth = classifiers_roc_curve(data, labels, DecisionTreeClassifier, "entropy" ) #makes roc curve
    max_depth = max( learned_max_depth )  # Should be read from the printed depth from Gridserach CV printed above 

    classification_report_container = confusion_matrix_and_classification_report( DecisionTreeClassifier(max_depth=max_depth, criterion="entropy", random_state=42) )
    print(f"Classification report for DecisionTreeClassifier(entropy, max_depth={max_depth} ) is \n {classification_report_container}. " )


    # for RandomForestClassifier with criterion = "gini"
    learned_max_depth = classifiers_roc_curve(data, labels, RandomForestClassifier, "gini") #makes roc curve

    max_depth = max( learned_max_depth )  # Should be read from the printed depth from Gridserach CV printed above 

    classification_report_container = confusion_matrix_and_classification_report( RandomForestClassifier(max_depth=max_depth, criterion="gini", random_state=42, n_jobs=-1) )
    print(f"Classification report for RandomForestClassifier(gini, max_depth={max_depth} ) is \n {classification_report_container}. " )

    # for RandomForestClassifier with criterion = "entropy"
    learned_max_depth = classifiers_roc_curve(data, labels, RandomForestClassifier, "entropy") #makes roc curve

    max_depth = max( learned_max_depth )  # Should be read from the printed depth from Gridserach CV printed above 

    classification_report_container = confusion_matrix_and_classification_report( RandomForestClassifier(max_depth=max_depth, criterion="entropy", random_state=42, n_jobs=-1) )
    print(f"Classification report for RandomForestClassifier(entropy, max_depth={max_depth} ) is \n {classification_report_container}. " )
