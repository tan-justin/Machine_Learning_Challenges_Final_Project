import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

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




            