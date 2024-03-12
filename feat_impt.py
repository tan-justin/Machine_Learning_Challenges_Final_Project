import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def normalize(sd, mean, val):
    return (val - mean)/sd

class FeatImportance:

    def __init__(self, data):
        self.data = data
        self.wholeX = data.iloc[:,:-1]
        self.wholeY = data.iloc[:,-1]
        X = self.wholeX.copy()
        y = self.wholeY.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
        means = X_train.mean()
        stds = X_train.std()
        X_train_normalized = X_train.apply(lambda x: normalize(stds[x.name], means[x.name], x), axis=0)
        X_test_normalized = X_test.apply(lambda x: normalize(stds[x.name], means[x.name], x), axis=0)
        self.train = pd.concat([X_train_normalized, y_train], axis=1)
        self.test = pd.concat([X_test_normalized, y_test], axis=1)

    def check_if_same(self):

        data = self.data.copy()
        X_train_normalized = self.train.copy().iloc[:,:-1]
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
        means = X_train.mean()
        stds = X_train.std()
        mean = {}
        std = {}
        for cl in X_train.columns:
            mean[cl] = X_train[cl].mean()
            std[cl] = X_train[cl].std()
        X_train_normalize = pd.DataFrame(index = X_train.index)
        for cl in X_train.columns:
            X_train_normalize[cl] = normalize(std.get(cl), mean.get(cl), X_train[cl])
        for cl in X_train.columns:
            print((X_train_normalized[cl] == X_train_normalize[cl]).unique())

    def random_forest_importance(self):

        X_train = self.train.copy().iloc[:,:-1]
        y_train = self.train.copy().iloc[:,-1]
        X_test = self.test.copy().iloc[:,:-1]
        y_test = self.test.copy().iloc[:,-1]

        X_whole = self.wholeX.copy()
        y_whole = self.wholeY.copy()

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        feature_importances = model.feature_importances_

        plt.figure(figsize=(32, 20))
        plt.bar(X_train.columns, feature_importances)
        plt.ylabel('Importance', fontsize = 20)
        plt.title('Feature Importances')
        plt.xticks(rotation=0, fontsize=15)
        #plt.show()

        directory = "Feature_Importance"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        classifier_name = type(model).__name__
        plt.savefig(f"{directory}/{classifier_name}_Feature_Importance.png")

        model2 = RandomForestClassifier()
        model2.fit(X_whole, y_whole)
        feature_importances2 = model2.feature_importances_

        iter = 0

        for cl in X_whole.columns:
            print(f'{cl}: {feature_importances[iter]}//{feature_importances2[iter]}')
            iter += 1

        plt.figure(figsize=(32, 20))
        plt.bar(X_whole.columns, feature_importances2)
        plt.ylabel('Importance', fontsize = 20)
        plt.title('Feature Importances')
        plt.xticks(rotation=0, fontsize = 15)

        classifier_name = type(model).__name__
        plt.savefig(f"{directory}/{classifier_name}_Feature_Importance_whole.png")


            
            

