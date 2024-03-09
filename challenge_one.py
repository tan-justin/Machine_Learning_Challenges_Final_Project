import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier #model 1
from sklearn.dummy import DummyClassifier #baseline
from sklearn.neighbors import KNeighborsClassifier #model 2
from sklearn.neural_network import MLPClassifier #model 3
#from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier #model 4 if we need it
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split #to generate train and test sets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

'''
In here, we're supposed to work on class imbalance. I need to take time to plan how to do challenge 2 and 3. 
1) Oversampling
2) SMOTE
3) Inverse Prior Probabilities

collinearity: feature selection, regularization, dimensionality reduction
outlier: removal of outliers, winsorizing, imputation
'''

def normalize(sd, mean, val):
    return (val - mean)/sd

class ChallengeOne:

    def __init__(self, data) -> None:
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
        self.oversampled = None
        self.smote = None
        self.ipp = None    

    def oversample(self):
        train = self.train.copy()
        feature_labels_name = train.columns
        class_count = Counter(train.iloc[:,-1])
        max_class_count = max(class_count.values())
        balanced = pd.DataFrame(columns = feature_labels_name)
        for class_label in class_count.keys():
            class_samples = train[train.iloc[:, -1] == class_label]
            num_samples_to_add = max_class_count - class_count[class_label]
            if num_samples_to_add > 0:
                random_samples = class_samples.sample(n = num_samples_to_add, replace = True) #get random samples to add
                balanced = pd.concat([balanced, random_samples])
            balanced = pd.concat([balanced, class_samples])
        object_columns = balanced.select_dtypes(include = ['object']).columns
        balanced[object_columns] = balanced[object_columns].astype('int64') #required because concatenation converts int64 to object
        self.oversampled = balanced.sample(frac = 1).reset_index(drop = True)

    def inverse_prior_probabilities(self):
        train = self.train.copy()
        class_counts = train.iloc[:, -1].value_counts(normalize = True)
        prior_probabilities = 1/(class_counts)
        #print(prior_probabilities)
        inverse_weights = prior_probabilities[train.iloc[:, -1]]
        weighted_features = train.iloc[:, :-1] * inverse_weights[:, np.newaxis]
        weighted_df = pd.concat([
            pd.DataFrame(weighted_features, columns = train.columns[:-1]),
            train.iloc[:, -1]
        ], axis=1)
        self.ipp = weighted_df
        #class_counts = self.ipp.iloc[:, -1].value_counts()
        #print("self.ipp:")
        #print(self.ipp)

    def apply_smote(self):
        train = self.train.copy()
        class_distribution = train.iloc[:,-1].value_counts(normalize = True) * 100
        class_labels = class_distribution.index.tolist()
        class_percentages = class_distribution.values.tolist()
        smote = SMOTE(random_state = 0)
        num_instances = len(train)
        target = []
        for label, percentage in zip(class_labels, class_percentages):
            num_instances_for_class = int(percentage * num_instances / 100)
            target.extend([label] * num_instances_for_class)
        train.iloc[:,-1] = target
        X_resampled, y_resampled = smote.fit_resample(train.iloc[:, :-1], train.iloc[:, -1])
        df_resampled = pd.DataFrame(X_resampled, columns = train.columns[:-1])
        df_resampled[train.columns[-1]] = y_resampled
        self.smote = df_resampled

    def test_and_print(self):
        #test each classifier using stratified k-fold cross validation on each dataset
        #Finally classify the test set with each classifier and get the accuracies
        oversampled = self.oversampled
        smote = self.smote
        ipp = self.ipp
        test = self.test

        X_o = oversampled.iloc[:,:-1]
        y_o = oversampled.iloc[:,-1]
        X_s = smote.iloc[:,:-1]
        y_s = smote.iloc[:,-1]
        X_i = ipp.iloc[:,:-1]
        y_i = ipp.iloc[:,-1]
        X_test = test.iloc[:,:-1]
        y_test = test.iloc[:,-1]

        skf = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 0)
        classifiers = ['rf','3nn','MLP','dummy']
        avg_o_accuracy = {}
        for classifier in classifiers:
            avg_o_accuracy[classifier] = 0

        for train_o, val_o in skf.split(X_o,y_o):
            X_train, X_val = X_o.iloc[train_o], X_o.iloc[val_o]
            y_train, y_val = y_o.iloc[train_o], y_o.iloc[val_o]
            for clf in classifiers:
                if clf == "rf":
                    rf = RandomForestClassifier(random_state = 0)
                    rf.fit(X_train, y_train)
                    y_pred_rf = rf.predict(X_val)
                    avg_o_accuracy[clf] += accuracy_score(y_val, y_pred_rf)
                
                if clf == '3nn':
                    nn3 = KNeighborsClassifier(n_neighbors = 3)
                    nn3.fit(X_train, y_train)
                    y_pred_3nn = nn3.predict(X_val)
                    avg_o_accuracy[clf] += accuracy_score(y_val, y_pred_3nn)
                
                if clf == 'MLP':
                    mlp = MLPClassifier(hidden_layer_sizes = (100, ), max_iter = 5000, random_state = 0)
                    mlp.fit(X_train, y_train)
                    y_pred_mlp = mlp.predict(X_val)
                    avg_o_accuracy[clf] += accuracy_score(y_val, y_pred_mlp)
                
                if clf == "dummy":
                    dum = DummyClassifier(strategy = 'stratified', random_state = 0)
                    dum.fit(X_train, y_train)
                    y_pred_dum = dum.predict(X_val)
                    avg_o_accuracy[clf] += accuracy_score(y_val, y_pred_dum)

        for classifier in avg_o_accuracy:
            avg_o_accuracy[classifier] /= 8
        #------------------------------------------------------------------------
        avg_s_accuracy = {}
        for classifier in classifiers:
            avg_s_accuracy[classifier] = 0

        for train_s, val_s in skf.split(X_s,y_s):
            X_train, X_val = X_s.iloc[train_s], X_s.iloc[val_s]
            y_train, y_val = y_s.iloc[train_s], y_s.iloc[val_s]
            for clf in classifiers:
                if clf == "rf":
                    rf = RandomForestClassifier(random_state = 0)
                    rf.fit(X_train, y_train)
                    y_pred_rf = rf.predict(X_val)
                    avg_s_accuracy[clf] += accuracy_score(y_val, y_pred_rf)
                
                if clf == '3nn':
                    nn3 = KNeighborsClassifier(n_neighbors = 3)
                    nn3.fit(X_train, y_train)
                    y_pred_3nn = nn3.predict(X_val)
                    avg_s_accuracy[clf] += accuracy_score(y_val, y_pred_3nn)
                
                if clf == 'MLP':
                    mlp = MLPClassifier(hidden_layer_sizes = (100, ), max_iter = 5000, random_state = 0)
                    mlp.fit(X_train, y_train)
                    y_pred_mlp = mlp.predict(X_val)
                    avg_s_accuracy[clf] += accuracy_score(y_val, y_pred_mlp)
                
                if clf == "dummy":
                    dum = DummyClassifier(strategy = 'stratified', random_state = 0)
                    dum.fit(X_train, y_train)
                    y_pred_dum = dum.predict(X_val)
                    avg_s_accuracy[clf] += accuracy_score(y_val, y_pred_dum)

        for classifier in avg_s_accuracy:
            avg_s_accuracy[classifier] /= 8
        #------------------------------------------------------------------------
        avg_i_accuracy = {}
        for classifier in classifiers:
            avg_i_accuracy[classifier] = 0
            
        for train_i, val_i in skf.split(X_i,y_i):
            X_train, X_val = X_i.iloc[train_i], X_i.iloc[val_i]
            y_train, y_val = y_i.iloc[train_i], y_i.iloc[val_i]
            for clf in classifiers:
                if clf == "rf":
                    rf = RandomForestClassifier(random_state = 0)
                    rf.fit(X_train, y_train)
                    y_pred_rf = rf.predict(X_val)
                    avg_i_accuracy[clf] += accuracy_score(y_val, y_pred_rf)
                
                if clf == '3nn':
                    nn3 = KNeighborsClassifier(n_neighbors = 3)
                    nn3.fit(X_train, y_train)
                    y_pred_3nn = nn3.predict(X_val)
                    avg_i_accuracy[clf] += accuracy_score(y_val, y_pred_3nn)
                
                if clf == 'MLP':
                    mlp = MLPClassifier(hidden_layer_sizes = (100, ), max_iter = 5000, random_state = 0)
                    mlp.fit(X_train, y_train)
                    y_pred_mlp = mlp.predict(X_val)
                    avg_i_accuracy[clf] += accuracy_score(y_val, y_pred_mlp)

                if clf == "dummy":
                    dum = DummyClassifier(strategy = 'stratified', random_state = 0)
                    dum.fit(X_train, y_train)
                    y_pred_dum = dum.predict(X_val)
                    avg_i_accuracy[clf] += accuracy_score(y_val, y_pred_dum)

        for classifier in avg_i_accuracy:
            avg_i_accuracy[classifier] /= 8
        #------------------------------------------------------------------------
        test_accuracy = {}
        strategy = ['Oversample','SMOTE','IPP']
        
        for strat in strategy:
            rf = RandomForestClassifier(random_state = 0)
            nn3 = KNeighborsClassifier(n_neighbors = 3)
            mlp = MLPClassifier(hidden_layer_sizes = (100, ), max_iter = 5000, random_state = 0)
            dum = DummyClassifier(strategy = 'stratified', random_state = 0)
            if strat == 'Oversample':
                rf.fit(X_o, y_o)
                nn3.fit(X_o, y_o)
                mlp.fit(X_o, y_o)
                dum.fit(X_o, y_o)

                y_pred_rf = rf.predict(X_test)
                y_pred_3nn = nn3.predict(X_test)
                y_pred_mlp = mlp.predict(X_test)
                y_pred_dum = dum.predict(X_test)

                acc_dict = {}
                for clf in classifiers:
                    if clf == 'rf':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_rf)
                    if clf == '3nn':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_3nn)
                    if clf == 'MLP':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_mlp)
                    if clf == 'dummy':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_dum)
                
                test_accuracy[strat] = acc_dict

            if strat == 'SMOTE':
                rf.fit(X_s, y_s)
                nn3.fit(X_s, y_s)
                mlp.fit(X_s, y_s)
                dum.fit(X_s, y_s)

                y_pred_rf = rf.predict(X_test)
                y_pred_3nn = nn3.predict(X_test)
                y_pred_mlp = mlp.predict(X_test)
                y_pred_dum = dum.predict(X_test)

                acc_dict = {}
                for clf in classifiers:
                    if clf == 'rf':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_rf)
                    if clf == '3nn':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_3nn)
                    if clf == 'MLP':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_mlp)
                    if clf == 'dummy':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_dum)
                
                test_accuracy[strat] = acc_dict
            
            if strat == 'IPP':
                rf.fit(X_i, y_i)
                nn3.fit(X_i, y_i)
                mlp.fit(X_i, y_i)
                dum.fit(X_i, y_i)

                y_pred_rf = rf.predict(X_test)
                y_pred_3nn = nn3.predict(X_test)
                y_pred_mlp = mlp.predict(X_test)
                y_pred_dum = dum.predict(X_test)

                #print(y_pred_rf)
                #print(y_pred_3nn)
                #print(y_pred_mlp)
                #print(y_test)

                acc_dict = {}
                for clf in classifiers:
                    if clf == 'rf':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_rf)
                    if clf == '3nn':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_3nn)
                    if clf == 'MLP':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_mlp)
                    if clf == 'dummy':
                        acc_dict[clf] = accuracy_score(y_test, y_pred_dum)
                
                test_accuracy[strat] = acc_dict

        return avg_o_accuracy, avg_s_accuracy, avg_i_accuracy, test_accuracy


        
    


