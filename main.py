from DataProcessing import read_csv_data, DataPreparation
from problem_detection import CollinearDetect

def main():
    csv_file_path = "winequality-red.csv"
    data = read_csv_data(csv_file_path)
    prep_instance = DataPreparation(data)
    prep_instance.generate_histogram()
    prep_instance.generate_pdf()
    prep_instance.generate_pie_chart()
    collinear_instance = CollinearDetect(data)
    collinear_instance.plot_correlation_matrix()
    collinear_instance.calculate_eigenvalues()
    collinear_instance.plot_scatterplots()
    collinear_instance.pca()
    collinear_instance.calculate_regression_coefficients()



if __name__ == "__main__":
    main()