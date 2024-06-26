'''
Name: Justin Tan
Assignment: Final Project
Date: March 20 2024
File: outliers.py
'''

import os
import pandas as pd

'''
Type: Class
Name: OutlierDetection
Purpose: Determining whether outliers exist within the dataset, and how prevalent they are
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: __init__
Purpose: Pre-processing of data and directory creation to store findings
Parameters: Dataset (data)
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: count_outliers_by_feature
Purpose: Determine which items are outliers. Outliers are defined if the feature value they possess is smaller than the 25th
percentile - 1.5 * (75th percentile - 25th percentile) or larger than the 75th percentile + 1.5 * (75th percentile - 25th
percentile) of the feature. Output the outliers as a file
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: calculate_z_scores
Purpose: Calculate the z-score values of each item based on the feature it represents. If the z-score magnitude is larger than 3, 
it is immediately treated as an outlier. All outliers will be outputted as part of a file along with the feature they are an
outlier in
Parameters: None
'''

class OutlierDetection:

    def __init__(self, data):
        self.y = data.iloc[:, -1].copy()
        self.x = data.iloc[:, :-1].copy()
        self.feature_labels = self.x.columns.tolist()
        self.output_dir = "outliers"
        print("Creating directory 'outliers' ")
        os.makedirs(self.output_dir, exist_ok = True)

    def count_outliers_by_feature(self, threshold = 1.5):
        x = self.x
        outlier_counts_output_file = os.path.join(self.output_dir, "outlier_counts.txt")
        with open(outlier_counts_output_file, "w") as file:
            print('Saving outlier count by feature into "outlier" directory')
            for column in self.feature_labels:
                quartile_1 = x[column].quantile(0.25)
                quartile_3 = x[column].quantile(0.75)
                iqr = quartile_3 - quartile_1
                lower_bound = quartile_1 - threshold * iqr
                upper_bound = quartile_3 + threshold * iqr
                outliers = x[column][(x[column] < lower_bound) | (x[column] > upper_bound)]
                file.write(f"Feature: {column}, Outlier Count: {len(outliers)}\n")
    
    def calculate_z_scores(self, threshold = 3):
        z_scores_output_file = os.path.join(self.output_dir, "z_scores.txt")
        with open(z_scores_output_file, "w") as file:
            print('Saving z score calculation by feature into "outlier" directory')
            for column in self.feature_labels:
                z_scores = (self.x[column] - self.x[column].mean()) / self.x[column].std()
                outliers = z_scores[abs(z_scores) > threshold]
                file.write(f"Feature: {column}, Outliers (abs(Z-Score) > {threshold}):\n{outliers.to_string(index = True)}\n")
                file.write(f"Number of outliers for feature '{column}': {len(outliers)}\n")