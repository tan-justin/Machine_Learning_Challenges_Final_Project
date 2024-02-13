import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import os

class CollinearDetect:

    def __init__(self, data):
        self.data = data
        self.y = data.iloc[:, -1].copy()
        self.x = data.iloc[:, :-1].copy()
        self.feature_labels = self.x.columns.tolist()
        self.output_dir = "collinear"
        print("Creating directory 'collinear' ")
        os.makedirs(self.output_dir, exist_ok = True)

    def plot_correlation_matrix(self):
        corr_matrix = self.x.copy().corr()
        plt.figure(figsize = (12, 10))
        plt.imshow(corr_matrix, cmap = 'coolwarm', interpolation = 'nearest')
        plt.colorbar()
        plt.title('Correlation Matrix')
        if self.feature_labels:
            plt.xticks(np.arange(len(corr_matrix.columns)), self.feature_labels, rotation = 90)
            plt.yticks(np.arange(len(corr_matrix.columns)), self.feature_labels)
        else:
            plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation = 90)
            plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
        print('Saving correlation heatmap into "collinear" directory')
        plt.savefig(os.path.join(self.output_dir, "correlation_matrix.png"))

    def calculate_eigenvalues(self):
        corr_matrix = self.x.corr()
        eigenvalues = np.linalg.eigvals(corr_matrix)
        output_file = os.path.join(self.output_dir, "eigenvalues.txt")
        with open(output_file, "w") as file:
            print('Saving eigen values into "collinear" directory')
            file.write("Eigenvalues of the correlation matrix:\n")
            for i, eigenvalue in enumerate(eigenvalues):
                if self.feature_labels:
                    file.write(f"{self.feature_labels[i]}: {eigenvalue}\n")
                else:
                    file.write(f"Feature {i}: {eigenvalue}\n")

    def plot_scatterplots(self):
        x = self.x.copy()
        num_features = len(x.columns)
        fig, axes = plt.subplots(num_features, num_features, figsize = (15, 15))
        for i in range(num_features):
            for j in range(num_features):
                if i != j:
                    axes[i, j].scatter(x.iloc[:, i], x.iloc[:, j])
                    axes[i, j].set_xlabel(x.columns[i])
                    axes[i, j].set_ylabel(x.columns[j])
                if self.feature_labels:
                    axes[i, j].set_xlabel(self.feature_labels[i])
                    axes[i, j].set_ylabel(self.feature_labels[j])
        plt.tight_layout()
        print('Saving scatterplots into "collinear" directory')
        plt.savefig(os.path.join(self.output_dir, "scatterplots.png"))

    def pca(self):
        pca = PCA()
        pca.fit(self.x)
        explained_variance_ratio = pca.explained_variance_ratio_
        output_file = os.path.join(self.output_dir, "explained_variance_ratio.txt")
        with open(output_file, "w") as file:
            print('Saving variance ratio into "collinear" directory')
            file.write("Explained variance ratio by each principal component:\n")
            for i, variance_ratio in enumerate(explained_variance_ratio):
                if self.feature_labels:
                    file.write(f"{self.feature_labels[i]}: {variance_ratio}\n")
                else:
                    file.write(f"Feature {i}: {variance_ratio}\n")

    def calculate_regression_coefficients(self):
        X = self.x 
        y = self.y
        model = LinearRegression()
        model.fit(X, y)
        coefficients = model.coef_
        output_file = os.path.join(self.output_dir, "regression_coefficients.txt")
        with open(output_file, "w") as file:
            print('Saving regression coefficients into "collinear" directory')
            file.write("Regression coefficients:\n")
            for i, coef in enumerate(coefficients):
                if self.feature_labels:
                    file.write(f"{self.feature_labels[i]}: {coef}\n")
                else:
                    file.write(f"Feature {i}: {coef}\n")