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

def normalize(sd, mean, val):
    return (val - mean)/sd

def oversample(train_set):
    train = train_set.copy()
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
    oversampled = balanced.sample(frac = 1).reset_index(drop = True)

    return oversampled
    
def cv_oversample(data_set, classifier):

    data  = data_set.copy()
    skf = StratifiedKFold(n_splits = 5, shuffle = 0, random_state = 0)
    X = data.copy().iloc[:,:-1]
    y = data.copy().iloc[:,:-1]
    avg_accuracy = {}
    for clf in classifier:
        avg_accuracy[clf] = 0
    for train, test in skf.split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        for clf in classifier:
            if clf == 'rf':
                rf = RandomForestClassifier(random_state = 0)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                avg_accuracy[clf] += accuracy_score(y_test, y_pred_rf)
            if clf == 'nn':
                nn = KNeighborsClassifier(n_neighbors = 3)
                nn.fit(X_train, y_train)
                y_pred_nn = nn.predict(X_test)
                avg_accuracy[clf] += accuracy_score(y_test, y_pred_nn)
            if clf == 'mlp':
                mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=5000, random_state=0)
                mlp.fit(X_train, y_train)
                y_pred_mlp = mlp.predict(X_test)
                avg_accuracy[clf] += accuracy_score(y_test, y_pred_mlp)
            if clf == 'dummy':
                dummy = DummyClassifier(strategy='stratified',random_state=0)
                dummy.fit(X_train, y_train)
                y_pred_dum = dummy.predict(X_test)
                avg_accuracy[clf] += accuracy_score(y_test, y_pred_dum)

    return avg_accuracy


def inverse_prior_probabilities(train_set):
    train = train_set.copy()
    class_counts = train.iloc[:, -1].value_counts(normalize = True)
    prior_probabilities = 1/(class_counts)
    inverse_weights = prior_probabilities[train.iloc[:, -1]]
    weighted_features = train.iloc[:, :-1] * inverse_weights[:, np.newaxis]
    weighted_df = pd.concat([
        pd.DataFrame(weighted_features, columns = train.columns[:-1]),
        train.iloc[:, -1]
    ], axis=1)
    ipp = weighted_df

def apply_smote(train_set):
    train = train_set.copy()
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
    smote = df_resampled


class ChallengeOneExtra:

    def __init__(self, data) -> None:
        self.data = data
        X = data.iloc[:,:-1]
        self.X = X
        y = data.iloc[:,-1]
        self.y = y
        self.train = None
        self.test = None
        self.oversampled = None
        self.smote = None
        self.ipp = None
        self.strategy = ['oversample','smote','ipp']
        self.classifier = ['rf','nn','mlp','dummy']

    def start(self):
        X = self.X
        y = self.y
        skf = StratifiedKFold(n_splits = 5, shuffle = 0, random_state = 0)
        
        for train, test in skf.split(X, y):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            self.train = pd.concat([X_train, y_train], axis = 1)
            self.test = pd.concat([X_test, y_test], axis = 1)

    

    

        