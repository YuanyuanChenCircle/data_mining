import pandas as pd
import numpy as np
import time
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
from sklearn.model_selection import learning_curve

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from ast import literal_eval


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import operator


from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import operator
import math

# need refractor here
classifierFunction = {
        "LogisticRegression": LogisticRegression,
        "KNN": KNeighborsClassifier,
        "SVC": SVC,
        "GradientBoosting": GradientBoostingClassifier,
        "DecisionTree": tree.DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "MLP": MLPClassifier,
        "GaussianNB": GaussianNB,
    }

METHODDICT = {
    "LogisticRegression": 'LogisticRegression',
    "KNN": "KNeighborsClassifier",
    "SVC": "SVC",
    "GradientBoosting": "GradientBoostingClassifier",
    "DecisionTree": "DecisionTreeClassifier",
    "RandomForest": "RandomForestClassifier",
    "MLP": "MLPClassifier",
    "GaussianNB": "GaussianNB",
}

def plot_learning_curve(classifier, df, label, cv = 5):
    i = math.floor(len(df) / 3)

    j = math.floor((len(df) / 3) * 2)
    k = math.floor((len(df) / 5) * 4)

    print(i)
    print(j)
    print(k)
    train_sizes, train_scores, valid_scores = learning_curve(classifier, df, label, train_sizes=[i, j, k], cv=5)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")


    plt.legend(loc="best")
    plt.savefig('./polls/static/polls/images/digits_CF_learning_curve.png', dpi=120)


def do_classification(df, label, df_droped_label, X_train, Y_train, X_test, Y_test, classifier_name,param):
    t_start = time.process_time()
    param_literal = json.loads(param)

    print(type(param_literal))
    classifier = classifierFunction[classifier_name](**param_literal)



    # classifier = classifier_dict[classifier_name]
    classifier.fit(X_train, Y_train)
    t_end = time.process_time()
    t_diff = t_end - t_start

    train_score = classifier.score(X_train, Y_train)
    test_score = classifier.score(X_test, Y_test)

    y_pred = classifier.predict(X_test)
    print(train_score)
    print(test_score)
    test_accuracry = metrics.precision_score(Y_test, y_pred, average='macro')
    print("precision")
    print(test_accuracry)
    recall = metrics.recall_score(Y_test, y_pred, average='macro')
    print("recall")
    print(recall)
    f1 = metrics.f1_score(Y_test, y_pred, average='macro')
    print("f1")
    print(f1)



    result = {'Classifier': METHODDICT[classifier_name], 'TrainPrecision': train_score, 'TestPrecision': test_score,
                                        'train_time': str(t_diff)+"s"}
    predict_label = classifier.predict(df_droped_label)
    i = math.floor(len(predict_label)/3)

    j = math.floor((len(predict_label)/3) * 2)
    k = math.floor((len(predict_label)/5) * 4)

    print(i)
    print(j)
    print(k)

    plot_learning_curve(classifier, df, label, cv=5)



    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(df)
    X_pca = PCA(n_components=2).fit_transform(df)

    ckpt_dir = "images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    plt.figure(figsize=(10, 5))

    print("#################")
    print(X_tsne.shape)
    print(X_tsne[:, 0].shape)
    # print(X_tsne[:,1])
    print(predict_label.shape)

    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=predict_label, label="t-SNE")
    plt.legend()
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=predict_label, label="PCA")
    plt.legend()

    plt.savefig('./polls/static/polls/images/digits_cf_data_view.png', dpi=120)
    # plt.show()



    return predict_label, result

"""
    This function fetches the training data set 
    and the origin dataset
"""
def get_train_test_data(df, label_name, x_name, ratio):
    mask = np.random.choice([True, False], size=len(df), p=[ratio, 1-ratio])

    df_train = df[mask]
    df_test = df[~mask]

    label = df[label_name].values

    label_train = df_train[label_name].values
    label_test = df_test[label_name].values
    x_train = df_train[x_name].values
    x_test = df_test[x_name].values
    print("class")
    print(len(df))

    return df, label, df_train, df_test, x_train, label_train, x_test, label_test

def MyClassification(df, label_name, classifier_name, train_ratio, param=""):
    train_ratio = float(train_ratio)
    x_name = list(df.columns.values)

    x_name.remove(label_name)
    df,label,df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test_data(df, label_name, x_name, train_ratio)

    predict_label, train_result = do_classification(df, label, df.drop([label_name], axis=1),X_train, Y_train, X_test, Y_test, classifier_name,param=param)
    new_df = df.copy(deep=True)
    new_df = new_df.rename(columns = {label_name: "Label"})
    new_df["prediction_result"] = predict_label
    print(train_result)
    print(predict_label)
    print(type(predict_label))
    print(len(predict_label))






    return new_df, train_result