'''
Name: Justin Tan
Assignment: Final Project
Date: March 20 2024
File: DataProcessing.py
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

'''
Type: Function
Name: read_csv_data
Purpose: Converts a csv dataset into a pandas dataframe
Parameters: file path of the data set
---------------------------------------------------------------------------------------------------------------------------------
Type: Class
Name: DataPreparation
Purpose: Obtain a data profile of the dataset
Parameters: dataset (data)
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: __init__
Purpose: setting the dataframe as self value
Parameters: Dataset
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: data_profile
Purpose: Obtain the following of each feature: mean, median, min value, max value, number of missing values and data type. Returns
a dictionary containing all of those values
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: generate_histogram
Purpose: Generate a histogram of each non-target feature showing the frequency of each value
Parameters: number of bins (10) and the output directory name
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: generate_pdf
Purpose: Generates a pdf data profile containing the information obtained from data_profile(), generate_histogram() and 
generate_pie_chart()
Parameters: output directory name
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: generate_pie_chart
Purpose: Create a pie chart showcasing the class distribution of the dataset. The class distribution is based on the target variable
Parameters: output directory name
'''

def read_csv_data(file_path):

    data = pd.read_csv(file_path)
    return data

class DataPreparation:

    def __init__(self, data) -> None:
        self.data = data

    def data_profile(self):

        data = self.data.copy()
        feature_labels_name = data.columns.tolist()
        class_labels_name = feature_labels_name.pop(-1)
        num_features = len(feature_labels_name)

        stat_dict = {}
        for columns in feature_labels_name:

            mean = data[columns].mean()
            median = data[columns].median()
            min = data[columns].min()
            max = data[columns].max()
            missing_values = data[columns].isnull().sum()

            stat_dict[columns] = {

                'type': data[columns].dtype,
                'mean': mean,
                'median': median,
                'min': min,
                'max': max,
                'number of missing values': missing_values
                            
            }

        return stat_dict
    
    def generate_histogram(self, bins = 10, output_path = 'histogram'):

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        data = self.data.copy()
        for columns in data.columns:

            data_without_null = data[columns].dropna()
            plt.hist(data_without_null, bins = bins, edgecolor = 'black')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title(f'Histogram for {columns}')
            output_file = os.path.join(output_path, f'{columns}_histogram.png')
            plt.savefig(output_file)
            plt.close()
            print(f'Histogram saved at: {output_file}')
    
    def generate_pdf(self, output_path='data_profile.pdf'):

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        try:

            for column, stats in self.data_profile().items():

                pdf.drawString(15, 750, f"Feature: {column}")
                pdf.drawString(15, 730, f"Type: {stats['type']}")
                pdf.drawString(15, 710, f"Mean: {stats['mean']}")
                pdf.drawString(15, 690, f"Median: {stats['median']}")
                pdf.drawString(15, 670, f"Max: {stats['max']}")
                pdf.drawString(15, 650, f"Min: {stats['min']}")
                pdf.drawString(15, 630, f"Missing Values: {stats['number of missing values']}")

                histogram_path = os.path.join('histogram', f'{column}_histogram.png')
                pdf.drawInlineImage(histogram_path, 10, 200, width=480, height=360)

                pdf.showPage()

        finally:
            pdf.save()

        buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(buffer.read())

        print(f'Combined PDF report saved at: {output_path}')

    def generate_pie_chart(self, output_path = 'pie chart'):

        data = self.data.copy()
        feature_labels_name = data.columns.tolist()
        class_labels_name = feature_labels_name.pop(-1)

        if not os.path.exists(output_path):

            os.makedirs(output_path)
        
        class_data = data[class_labels_name]
        class_instance_count = class_data.value_counts()
        plt.pie(class_instance_count, labels = class_instance_count.index, startangle=90)
        labels = [f'{category}: {count / class_instance_count.sum() * 100:.1f}%' for category, count in class_instance_count.items()]
        plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title('Categorical Distribution of Wine Quality')
        output_file = os.path.join(output_path, f'{class_labels_name}_pie_chart.png')
        plt.savefig(output_file)
        plt.close()
        print(f'Pie chart saved at: {output_file}')



            