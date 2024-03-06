import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier #model 1
from sklearn.dummy import DummyClassifier #baseline
from sklearn.neighbors import KNeighborsClassifier #model 2
from sklearn.neural_network import MLPClassifier #model 3
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier #model 4 if we need it
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split #to generate train and test sets

'''
In here, we're supposed to work on class imbalance. I need to take time to plan how to do challenge 2 and 3. 
1) Oversampling
2) Random Sampling (using differnet fold strategies)
3) Inverse Prior Probabilities

collinearity: feature selection, regularization, dimensionality reduction
outlier: removal of outliers, winsorizing, imputation
'''

class ChallengeOne:

    def __init__(self, data) -> None:
        self.data = data
        X = data[:,:-1]
        y = data[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        self.train = pd.concat([X_train, y_train], axis = 1)
        self.test = pd.concat([X_test, y_test], axis = 1)
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
        #feature_labels_name = data.columns
        prior_probabilities = np.ones(len(train)) / len(train)
        inverse_weights = 1 / prior_probabilities
        weighted_features = train.iloc[:, :-1] * inverse_weights[:,np.newaxis]
        weighted_target = train.iloc[:, -1] * inverse_weights
        weighted_df = pd.concat([
            pd.DataFrame(weighted_features, columns = train.columns[:-1]), 
            pd.DataFrame(weighted_target, columns = [train.columns[-1]])
        ], axis=1)
        self.ipp = weighted_df

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

