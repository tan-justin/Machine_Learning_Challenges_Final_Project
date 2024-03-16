'''
Name: Justin Tan
Assignment: Final Project
Date: March 20 2024
File: challenge_two.py
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
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

'''
Type: Function
Name: normalize
Purpose: perform normalization for dataset to handle 
Parameters: standard deviation (sd), mean (mean), value to be normalized (val)
---------------------------------------------------------------------------------------------------------------------------------
Type: Class
Name: ChallengeTwo
Purpose: Performs pre-processing of data and contain the functions required to apply multicollinearity strategies and test them
Parameters: dataset (data)
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: __init__
Purpose: Pre-processing of data, train-test splits and normalization of data
Parameters: Dataset
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: feat_select
Purpose: Performs VIF on the dataset to determine which features to drop and create a new dataset that has those features dropped,
then using the new dataset, perform cross-validation
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: dimen_reduction
Purpose: Make a copy of the dataset and apply PCA onto the dataset. Perform cross-validation on this dataset that has PCA applied
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: regularization
Purpose: Apply a regularization parameter to the logistic regression classifier and apply an alpha parameter to the MLP
classifier. Random Forest has new parameters to perform early stopping, which is similar to regularization
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: test_acc
Purpose: Perform train-test using models trained on strategy-applied datasets and for regularization, using the same models in 
the regularization function
Parameters: None
'''

def normalize(sd, mean, val):
    return (val - mean)/sd

class ChallengeTwo:
    def __init__(self, data) -> None:
        self.data = data
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        means = X_train.mean()
        stds = X_train.std()
        X_train_normalized = X_train.apply(lambda x: normalize(stds[x.name], means[x.name], x), axis=0)
        X_test_normalized = X_test.apply(lambda x: normalize(stds[x.name], means[x.name], x), axis=0)
        self.train = pd.concat([X_train_normalized, y_train], axis=1)
        self.test = pd.concat([X_test_normalized, y_test], axis=1)
        self.regularize = None
        self.vif = None
        self.pca = None
        self.feat_drop_select = None
        self.feat_drop_pca = None

    def feat_select(self):
        train_f = self.train.copy()
        X = train_f.iloc[:,:-1]
        y = train_f.iloc[:,-1]
        X = sm.add_constant(X)
        vif = pd.DataFrame()
        vif["Features"] = X.columns
        vif["VIF Factor"] = [
            1 / (1 - sm.OLS(X.iloc[:, i], X.iloc[:, :i] if i > 0 else X.iloc[:, 1:]).fit().rsquared) 
            for i in range(X.shape[1])
        ]
        threshold = 2
        features_to_drop = []

        for index, row in vif.iterrows():
            if row['VIF Factor'] > threshold:
                features_to_drop.append(row['Features'])
        X = X.drop(columns = features_to_drop)
        X = X.drop(columns = ['const'])
        #print(features_to_drop)
        #print(X)
        concatenated_data = pd.concat([X, y], axis=1)
        #print(concatenated_data)
        self.vif = concatenated_data
        self.feat_drop_select = features_to_drop
        skf = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 0)
        classifiers = ['rf','LR','MLP','dummy']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train, val in skf.split(X, y):
            X_train, X_val = X.iloc[train], X.iloc[val]
            y_train, y_val = y.iloc[train], y.iloc[val]
            for clf in classifiers:
                if clf == 'rf':
                    rf = RandomForestClassifier(random_state = 0)
                    rf.fit(X_train, y_train)
                    y_pred_rf = rf.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_rf)
                if clf == 'LR':
                    lr = LogisticRegression(penalty = None, solver = 'lbfgs', max_iter = 1000, random_state = 0)
                    lr.fit(X_train, y_train)
                    y_pred_lr = lr.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_lr)
                if clf == 'MLP':
                    mlp = MLPClassifier(hidden_layer_sizes = (100,), max_iter = 5000, random_state = 0)
                    mlp.fit(X_train, y_train)
                    y_pred_mlp = mlp.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_mlp)
                if clf == 'dummy':
                    dummy = DummyClassifier(strategy = 'stratified', random_state = 0)
                    dummy.fit(X_train, y_train)
                    y_pred_dummy = dummy.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_dummy)
                #print(avg_accuracy)

        for clf in avg_accuracy:
            avg_accuracy[clf] /= 8
            
        return avg_accuracy
            
    
    def dimen_reduction(self):
        train_d = self.train.copy()
        X = train_d.iloc[:,:-1]
        y = train_d.iloc[:,-1]
        pca = PCA()
        pca.fit(X)
        self.pca = pca
        #print(X_pca_df)
        skf = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 0)
        classifiers = ['rf','LR','MLP','dummy']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train, val in skf.split(X, y):
            X_train, X_val = X.iloc[train], X.iloc[val]
            y_train, y_val = y.iloc[train], y.iloc[val]
            for clf in classifiers:
                if clf == 'rf':
                    rf = RandomForestClassifier(random_state = 0)
                    rf.fit(X_train, y_train)
                    y_pred_rf = rf.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_rf)
                if clf == 'LR':
                    lr = LogisticRegression(penalty = None, solver = 'lbfgs', max_iter = 1000, random_state = 0)
                    lr.fit(X_train, y_train)
                    y_pred_lr = lr.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_lr)
                if clf == 'MLP':
                    mlp = MLPClassifier(hidden_layer_sizes = (100,), max_iter = 5000, random_state = 0)
                    mlp.fit(X_train, y_train)
                    y_pred_mlp = mlp.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_mlp)
                if clf == 'dummy':
                    dummy = DummyClassifier(strategy = 'stratified', random_state = 0)
                    dummy.fit(X_train, y_train)
                    y_pred_dummy = dummy.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_dummy)

        for clf in avg_accuracy:
            avg_accuracy[clf] /= 8
            
        return avg_accuracy
        
    def regularization(self):
        train_r = self.train.copy()
        X = train_r.iloc[:,:-1]
        y = train_r.iloc[:,-1]
        skf = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 0)
        classifiers = ['rf','LR','MLP','dummy']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train, val in skf.split(X, y):
            X_train, X_val = X.iloc[train], X.iloc[val]
            y_train, y_val = y.iloc[train], y.iloc[val]
            for clf in classifiers:
                if clf == 'rf':
                    rf = RandomForestClassifier(max_depth = 10, min_samples_leaf = 2, min_samples_split = 5, random_state = 0, max_features = 8)
                    rf.fit(X_train, y_train)
                    y_pred_rf = rf.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_rf)
                if clf == 'LR':
                    lr = LogisticRegression(penalty = 'l2', solver = 'lbfgs', max_iter = 1000, random_state = 0)
                    lr.fit(X_train, y_train)
                    y_pred_lr = lr.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_lr)
                if clf == 'MLP':
                    mlp = MLPClassifier(hidden_layer_sizes = (100,), max_iter = 5000, random_state = 0, alpha = 0.001)
                    mlp.fit(X_train, y_train)
                    y_pred_mlp = mlp.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_mlp)
                if clf == 'dummy':
                    dummy = DummyClassifier(strategy = 'stratified', random_state = 0)
                    dummy.fit(X_train, y_train)
                    y_pred_dummy = dummy.predict(X_val)
                    avg_accuracy[clf] += accuracy_score(y_val, y_pred_dummy)

        for clf in avg_accuracy:
            avg_accuracy[clf] /= 8
            
        return avg_accuracy
        
    def test_acc(self):
        train_r = self.train.copy()
        test = self.test.copy()
        train_f = self.vif.copy()
        pca = self.pca
        feat_drop_select = self.feat_drop_select.copy()

        strategy = ['vif','dr','reg']
        test_accuracy = {}
        classifiers = ['rf','LR','MLP','dummy']

        for strat in strategy:
            if strat != 'reg':
                rf = RandomForestClassifier(random_state = 0)
                lr = LogisticRegression(penalty= None, solver='lbfgs', max_iter=1000, random_state = 0)
                mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter = 5000, random_state = 0)
                dummy = DummyClassifier(strategy = 'stratified', random_state = 0)
                if strat == 'vif':
                    accuracy = {}
                    X_train = train_f.iloc[:,:-1]
                    y_train = train_f.iloc[:,-1]
                    X_test_pre = test.iloc[:,:-1]
                    y_test = test.iloc[:,-1]
                    X_test = X_test_pre.drop(columns = feat_drop_select)

                    rf.fit(X_train, y_train)
                    lr.fit(X_train, y_train)
                    mlp.fit(X_train, y_train)
                    dummy.fit(X_train, y_train)

                    y_pred_rf = rf.predict(X_test)
                    y_pred_lr = lr.predict(X_test)
                    y_pred_mlp = mlp.predict(X_test)
                    y_pred_dummy = dummy.predict(X_test)

                    for clf in classifiers:
                        if clf == 'rf':
                            accuracy[clf] = accuracy_score(y_test, y_pred_rf)
                        if clf == 'LR':
                            accuracy[clf] = accuracy_score(y_test, y_pred_lr)
                        if clf == 'MLP':
                            accuracy[clf] = accuracy_score(y_test, y_pred_mlp)
                        if clf == 'dummy':
                            accuracy[clf] = accuracy_score(y_test, y_pred_dummy)
                    test_accuracy[strat] = accuracy
                
                if strat == 'dr':
                    accuracy = {}
                    X_train = pca.transform(train_r.iloc[:,:-1])
                    y_train = train_r.iloc[:,-1]
                    X_test = pca.transform(test.iloc[:,:-1])
                    y_test = test.iloc[:,-1]

                    rf.fit(X_train, y_train)
                    lr.fit(X_train, y_train)
                    mlp.fit(X_train, y_train)
                    dummy.fit(X_train, y_train)

                    y_pred_rf = rf.predict(X_test)
                    y_pred_lr = lr.predict(X_test)
                    y_pred_mlp = mlp.predict(X_test)
                    y_pred_dummy = dummy.predict(X_test)

                    for clf in classifiers:
                        if clf == 'rf':
                            accuracy[clf] = accuracy_score(y_test, y_pred_rf)
                        if clf == 'LR':
                            accuracy[clf] = accuracy_score(y_test, y_pred_lr)
                        if clf == 'MLP':
                            accuracy[clf] = accuracy_score(y_test, y_pred_mlp)
                        if clf == 'dummy':
                            accuracy[clf] = accuracy_score(y_test, y_pred_dummy)
                    test_accuracy[strat] = accuracy

            if strat == 'reg':
                rf = RandomForestClassifier(max_depth = 10, min_samples_leaf = 2, min_samples_split=5, random_state = 0, max_features = 8) 
                #similar to regularization, there is no regularization parameter for RF classifiers. 
                lr = LogisticRegression(penalty= 'l2', solver='lbfgs', max_iter=1000, random_state = 0)
                mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter = 5000, random_state = 0, alpha = 0.001) #alpha multiplied by 10
                dummy = DummyClassifier(strategy = 'stratified', random_state = 0)

                accuracy = {}
                X_train = train_r.iloc[:,:-1]
                y_train = train_r.iloc[:,-1]
                X_test = test.iloc[:,:-1]
                y_test = test.iloc[:,-1]

                rf.fit(X_train, y_train)
                lr.fit(X_train, y_train)
                mlp.fit(X_train, y_train)
                dummy.fit(X_train, y_train)

                y_pred_rf = rf.predict(X_test)
                y_pred_lr = lr.predict(X_test)
                y_pred_mlp = mlp.predict(X_test)
                y_pred_dummy = dummy.predict(X_test)

                for clf in classifiers:
                    if clf == 'rf':
                        accuracy[clf] = accuracy_score(y_test, y_pred_rf)
                    if clf == 'LR':
                        accuracy[clf] = accuracy_score(y_test, y_pred_lr)
                    if clf == 'MLP':
                        accuracy[clf] = accuracy_score(y_test, y_pred_mlp)
                    if clf == 'dummy':
                        accuracy[clf] = accuracy_score(y_test, y_pred_dummy)
                test_accuracy[strat] = accuracy

        return test_accuracy
            


            





