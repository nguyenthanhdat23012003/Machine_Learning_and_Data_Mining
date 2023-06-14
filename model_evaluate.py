from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def model_evaluate(model, X_train, y_train, X_test, y_test):
    acc_train = model.score(X_train, y_train)
    print('Accuracy of model on training data : {}%'.format(acc_train*100))
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy of model on testing data: {}%".format(
        accuracy_score(y_test, y_pred) * 100))
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative', 'Positive', 'Litigious', 'Uncertainly']
    group_names = ['True Neg', 'False Pos', 'False Lit', 'False Uncer',
                   'False Neg', 'True Pos', 'False Lit', 'False Uncer',
                   'False Neg', 'False Pos', 'True Lit', 'False Uncer',
                   'False Neg', 'False Pos', 'False Lit', 'True Uncer']
    group_percentages = ['{0:.2%}'.format(
        value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(4, 4)

    sns.heatmap(cf_matrix, annot=labels, cmap='Reds', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
