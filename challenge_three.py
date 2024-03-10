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

        #print(X)
        self.winsor = pd.concat([X, y], axis = 1)
        #print(self.winsor)
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
        
        train_original = self.train.copy()
        train = self.train.copy()
        X = train_original.iloc[:,:-1]
        y = train_original.iloc[:,-1]
        features = X.columns[:-1]
        target_column = 'quality'
        num_bins = 5
        for feature in X.columns:
            sorted_indices = np.argsort(train[feature])
            bin_size = len(train) // num_bins
            remainder = len(train) % num_bins
            bins = np.zeros(len(train))
            for i in range(num_bins):
                bins[sorted_indices[i * bin_size : (i + 1) * bin_size]] = i
            if remainder > 0:
                bins[sorted_indices[num_bins * bin_size:]] = num_bins - 1
            train[f'{feature}_bin_equal_freq'] = bins.astype(int)
        target = train.pop(target_column)
        train[target_column] = target
        self.bins = train
        self.bin = bins

        X = train.iloc[:,:-1]
        y = train.iloc[:,-1]
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

        strategy = ['remove','winsor','imput']
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


            
            
        print(test_accuracy)

        




