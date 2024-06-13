'''
Name: Justin Tan
Assignment: Final Project
Date: March 20 2024
File: challenge_three.py
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier #model 1
from sklearn.dummy import DummyClassifier #baseline
from sklearn.neighbors import KNeighborsClassifier #model 2
from sklearn.neural_network import MLPClassifier #model 3
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split #to generate train and test sets
from sklearn.metrics import accuracy_score
from scipy.stats import zscore

'''
Type: Function
Name: normalize
Purpose: perform normalization for dataset to handle 
Parameters: standard deviation (sd), mean (mean), value to be normalized (val)
---------------------------------------------------------------------------------------------------------------------------------
Type: Class
Name: ChallengeThree
Purpose: Performs pre-processing of data and contain the functions required to apply outlier strategies and test them
Parameters: dataset (data)
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: __init__
Purpose: Pre-processing of data, train-test splits and normalization of data
Parameters: Dataset
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: remove_outliers
Purpose: Outlier items are items that have at least one feature with a z-score value that is bigger than 3 or smaller than -3.
A copy of the dataset will have those outlier items removed, and cross validation is performed on the new dataset
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: winsorize
Purpose: For outlier items that have a feature value that has a z-score value larger than 3 or smaller than -3, replace those
values with the 95th percentile or 5th percentile of the feature respectively. Cross-validation was performed on the new
dataset
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: imputation (modified winsorizing would be the correct terminology)
Purpose: Similar to winsorize, but instead of replacing with the 95th percentile or the 5th percentile, we replace outlier values
with the median of the respective feature. Cross-validation is performed on the new dataset 
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: Binning
Purpose: Perform binning of values for each feature. The binning boundaries are: (-inf, -3],(-3, 25th percentile],
(25th percentile, 50th percentile],(50th percentile, 75th percentile],(75th percentile, 3),[3, inf) 
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: test_acc
Purpose: Perform train-test using models trained on strategy-applied datasets
Parameters: None
'''

def normalize(sd, mean, val):
    return (val - mean)/sd

class ChallengeThree:

    def __init__(self, data):
        self.data = data
        features = data.columns[:-1]
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        means = X_train.mean()
        stds = X_train.std()
        X_train_normalized = X_train.apply(lambda x: normalize(stds[x.name], means[x.name], x), axis=0)
        X_test_normalized = X_test.apply(lambda x: normalize(stds[x.name], means[x.name], x), axis=0)
        self.train = pd.concat([X_train_normalized, y_train], axis=1)
        self.test = pd.concat([X_test_normalized, y_test], axis=1)
        self.remove = None
        self.winsor = None
        self.imput = None
        self.bin = None
        self.bins = None

    def remove_outliers(self):
        train = self.train.copy()
        X = train.iloc[:,:-1]
        y = train.iloc[:,-1]
        z_scores = (X - X.mean()) / X.std()
        outliers = (z_scores > 3) | (z_scores < -3)
        filtered_X = X[~outliers.any(axis = 1)]
        filtered_y = y[~outliers.any(axis = 1)]
        #print(filtered_X)
        #print(filtered_y)
        self.remove = pd.concat([filtered_X, filtered_y], axis = 1)

        # Cross-validation starts here

        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        classifiers = ['rf','3nn','MLP','dummy']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train, val in skf.split(filtered_X, filtered_y):
            X_train, X_val = X.iloc[train], X.iloc[val]
            y_train, y_val = y.iloc[train], y.iloc[val]
            for clf in classifiers:
                if clf == "rf":
                    rf = RandomForestClassifier(random_state = 0)
                    rf.fit(X_train, y_train)
                    y_pred_rf = rf.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_rf)
                if clf == "3nn":
                    nn = KNeighborsClassifier(n_neighbors = 3)
                    nn.fit(X_train, y_train)
                    y_pred_nn = nn.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_nn)
                if clf == "MLP":
                    mlp = MLPClassifier(hidden_layer_sizes = (100,), max_iter = 5000, random_state = 0)
                    mlp.fit(X_train, y_train)
                    y_pred_mlp = mlp.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_mlp)
                if clf == "dummy":
                    dummy = DummyClassifier(strategy = "stratified", random_state = 0)
                    dummy.fit(X_train, y_train)
                    y_pred_dummy = dummy.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_dummy)

        for clf in avg_accuracy:
            avg_accuracy[clf] /= 5

        print(avg_accuracy)

    def winsorize(self):
        train = self.train.copy()
        #print(train)
        X = train.iloc[:,:-1]
       #print(X)
        y = train.iloc[:,-1]
        features = X.columns[:-1]
        z_scores = X.apply(zscore)

        for col in X.columns:
            outliers_high = z_scores[col] > 3
            outliers_low = z_scores[col] < -3
            if outliers_high.any():
                X.loc[outliers_high, col] = X[col][~outliers_high].quantile(0.95)
            if outliers_low.any():
                X.loc[outliers_low, col] = X[col][~outliers_low].quantile(0.05)

        self.winsor = pd.concat([X, y], axis = 1)
        # Cross-validation starts here

        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        classifiers = ['rf','3nn','MLP','dummy']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train, val in skf.split(X, y):
            X_train, X_val = X.iloc[train], X.iloc[val]
            y_train, y_val = y.iloc[train], y.iloc[val]
            for clf in classifiers:
                if clf == "rf":
                    rf = RandomForestClassifier(random_state = 0)
                    rf.fit(X_train, y_train)
                    y_pred_rf = rf.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_rf)
                if clf == "3nn":
                    nn = KNeighborsClassifier(n_neighbors = 3)
                    nn.fit(X_train, y_train)
                    y_pred_nn = nn.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_nn)
                if clf == "MLP":
                    mlp = MLPClassifier(hidden_layer_sizes = (100,), max_iter = 5000, random_state = 0)
                    mlp.fit(X_train, y_train)
                    y_pred_mlp = mlp.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_mlp)
                if clf == "dummy":
                    dummy = DummyClassifier(strategy = "stratified", random_state = 0)
                    dummy.fit(X_train, y_train)
                    y_pred_dummy = dummy.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_dummy)

        for clf in avg_accuracy:
            avg_accuracy[clf] /= 5

        print(avg_accuracy)

    def imputation(self): 

        train = self.train.copy()
        X = train.iloc[:,:-1]
        y = train.iloc[:,-1]
        features = X.columns[:-1]
        z_scores = X.apply(zscore)
    
        outlier_items = X[(z_scores > 3) | (z_scores < -3)]
        for index, row in outlier_items.iterrows():
            for column in X.columns:
                if z_scores.at[index, column] > 3 or z_scores.at[index, column] < -3:
                    median = X[column].median()
                    X.at[index, column] = median

        self.imput = pd.concat([X, y], axis = 1)

        # Cross-validation starts here
        
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        classifiers = ['rf','3nn','MLP','dummy']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train, val in skf.split(X, y):
            X_train, X_val = X.iloc[train], X.iloc[val]
            y_train, y_val = y.iloc[train], y.iloc[val]
            for clf in classifiers:
                if clf == "rf":
                    rf = RandomForestClassifier(random_state = 0)
                    rf.fit(X_train, y_train)
                    y_pred_rf = rf.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_rf)
                if clf == "3nn":
                    nn = KNeighborsClassifier(n_neighbors = 3)
                    nn.fit(X_train, y_train)
                    y_pred_nn = nn.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_nn)
                if clf == "MLP":
                    mlp = MLPClassifier(hidden_layer_sizes = (100,), max_iter = 5000, random_state = 0)
                    mlp.fit(X_train, y_train)
                    y_pred_mlp = mlp.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_mlp)
                if clf == "dummy":
                    dummy = DummyClassifier(strategy = "stratified", random_state = 0)
                    dummy.fit(X_train, y_train)
                    y_pred_dummy = dummy.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_dummy)

        for clf in avg_accuracy:
            avg_accuracy[clf] /= 5

        print(avg_accuracy)

    def binning(self):
        
        train = self.train.copy()
        X = train.iloc[:,:-1]
        y = train.iloc[:,-1]
        
        z_scores = pd.DataFrame(index = X.index)

        for column in X.columns:
            z_score = (X[column] - X[column].mean())/X[column].std()
            z_scores[column] = z_score
        bin_boundaries = [-np.inf, -3, np.percentile(z_scores.values, 25), np.percentile(z_scores.values, 50),
                  np.percentile(z_scores.values, 75), 3, np.inf]
        binned_features = pd.DataFrame(index= X.index)
        for column in z_scores.columns:
            binned_features[f'{column}_bin'] = np.digitize(z_scores[column], bin_boundaries)
        binned_X = pd.concat([X, binned_features], axis=1)
        binned_df = pd.concat([binned_X, y], axis=1)
        self.bins = binned_df
        self.bin = bin_boundaries
        X = binned_df.iloc[:,:-1]
        y = binned_df.iloc[:,-1]

        # Cross-validation starts here

        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        classifiers = ['rf','3nn','MLP','dummy']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train, val in skf.split(X, y):
            X_train, X_val = X.iloc[train], X.iloc[val]
            y_train, y_val = y.iloc[train], y.iloc[val]
            for clf in classifiers:
                if clf == "rf":
                    rf = RandomForestClassifier(random_state = 0)
                    rf.fit(X_train, y_train)
                    y_pred_rf = rf.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_rf)
                if clf == "3nn":
                    nn = KNeighborsClassifier(n_neighbors = 3)
                    nn.fit(X_train, y_train)
                    y_pred_nn = nn.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_nn)
                if clf == "MLP":
                    mlp = MLPClassifier(hidden_layer_sizes = (100,), max_iter = 5000, random_state = 0)
                    mlp.fit(X_train, y_train)
                    y_pred_mlp = mlp.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_mlp)
                if clf == "dummy":
                    dummy = DummyClassifier(strategy = "stratified", random_state = 0)
                    dummy.fit(X_train, y_train)
                    y_pred_dummy = dummy.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_dummy)

        for clf in avg_accuracy:
            avg_accuracy[clf] /= 5

        print(avg_accuracy)


    def test_strat(self):

        test = self.test.copy()
        imput = self.imput.copy()
        winsor = self.winsor.copy()
        remove = self.remove.copy()
        bins = self.bins.copy()
        boundary = self.bin

        strategy = ['remove','winsor','imput','bin']
        test_accuracy = {}
        classifiers = ['rf','3nn','MLP','dummy']

        for strat in strategy:
            rf = RandomForestClassifier(random_state = 0)
            nn3 = KNeighborsClassifier(n_neighbors = 3)
            mlp = MLPClassifier(hidden_layer_sizes = (100, ), max_iter = 5000, random_state = 0)
            dum = DummyClassifier(strategy = 'stratified', random_state = 0)
            if strat == 'remove':
                acc = {}

                X_train = remove.iloc[:,:-1]
                y_train = remove.iloc[:,-1]
                X_test = test.iloc[:,:-1]
                y_test = test.iloc[:,-1]

                rf.fit(X_train, y_train)
                nn3.fit(X_train, y_train)
                mlp.fit(X_train, y_train)
                dum.fit(X_train, y_train)

                y_pred_rf = rf.predict(X_test)
                y_pred_nn3 = nn3.predict(X_test)
                y_pred_mlp = mlp.predict(X_test)
                y_pred_dum = dum.predict(X_test)

                for clf in classifiers:
                    if clf == 'rf':
                        acc[clf] = accuracy_score(y_test, y_pred_rf)
                    if clf == '3nn':
                        acc[clf] = accuracy_score(y_test, y_pred_nn3)
                    if clf == 'MLP':
                        acc[clf] = accuracy_score(y_test, y_pred_mlp)
                    if clf == 'dummy':
                        acc[clf] = accuracy_score(y_test, y_pred_dum)

                test_accuracy[strat] = acc

            if strat == 'winsor':
                acc = {}

                X_train = winsor.iloc[:,:-1]
                y_train = winsor.iloc[:,-1]
                X_test = test.iloc[:,:-1]
                y_test = test.iloc[:,-1]

                rf.fit(X_train, y_train)
                nn3.fit(X_train, y_train)
                mlp.fit(X_train, y_train)
                dum.fit(X_train, y_train)

                y_pred_rf = rf.predict(X_test)
                y_pred_nn3 = nn3.predict(X_test)
                y_pred_mlp = mlp.predict(X_test)
                y_pred_dum = dum.predict(X_test)

                for clf in classifiers:
                    if clf == 'rf':
                        acc[clf] = accuracy_score(y_test, y_pred_rf)
                    if clf == '3nn':
                        acc[clf] = accuracy_score(y_test, y_pred_nn3)
                    if clf == 'MLP':
                        acc[clf] = accuracy_score(y_test, y_pred_mlp)
                    if clf == 'dummy':
                        acc[clf] = accuracy_score(y_test, y_pred_dum)

                test_accuracy[strat] = acc
            
            if strat == 'imput':
                acc = {}

                X_train = imput.iloc[:,:-1]
                y_train = imput.iloc[:,-1]
                X_test = test.iloc[:,:-1]
                y_test = test.iloc[:,-1]

                rf.fit(X_train, y_train)
                nn3.fit(X_train, y_train)
                mlp.fit(X_train, y_train)
                dum.fit(X_train, y_train)

                y_pred_rf = rf.predict(X_test)
                y_pred_nn3 = nn3.predict(X_test)
                y_pred_mlp = mlp.predict(X_test)
                y_pred_dum = dum.predict(X_test)

                for clf in classifiers:
                    if clf == 'rf':
                        acc[clf] = accuracy_score(y_test, y_pred_rf)
                    if clf == '3nn':
                        acc[clf] = accuracy_score(y_test, y_pred_nn3)
                    if clf == 'MLP':
                        acc[clf] = accuracy_score(y_test, y_pred_mlp)
                    if clf == 'dummy':
                        acc[clf] = accuracy_score(y_test, y_pred_dum)

                test_accuracy[strat] = acc

            if strat == 'bin':
                X_train = bins.iloc[:,:-1]
                y_train = bins.iloc[:,-1]
                X = test.iloc[:,:-1]
                y_test = test.iloc[:,-1]
                acc = {}
                
                z_scores = pd.DataFrame(index = X.index)
                for column in X.columns:
                    z_score = (X[column] - X[column].mean())/X[column].std()
                    z_scores[column] = z_score
                binned_features = pd.DataFrame(index= X.index)
                for column in z_scores.columns:
                    binned_features[f'{column}_bin'] = np.digitize(z_scores[column], boundary)
                X_test = pd.concat([X, binned_features], axis=1)

                rf.fit(X_train, y_train)
                nn3.fit(X_train, y_train)
                mlp.fit(X_train, y_train)
                dum.fit(X_train, y_train)

                y_pred_rf = rf.predict(X_test)
                y_pred_nn3 = nn3.predict(X_test)
                y_pred_mlp = mlp.predict(X_test)
                y_pred_dum = dum.predict(X_test)

                for clf in classifiers:
                    if clf == 'rf':
                        acc[clf] = accuracy_score(y_test, y_pred_rf)
                    if clf == '3nn':
                        acc[clf] = accuracy_score(y_test, y_pred_nn3)
                    if clf == 'MLP':
                        acc[clf] = accuracy_score(y_test, y_pred_mlp)
                    if clf == 'dummy':
                        acc[clf] = accuracy_score(y_test, y_pred_dum)

                test_accuracy[strat] = acc

        print(test_accuracy)

        




