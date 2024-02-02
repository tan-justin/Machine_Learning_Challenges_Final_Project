from DataProcessing import read_csv_data,DataPreparation

def main():
    csv_file_path = "winequality-red.csv"
    data = read_csv_data(csv_file_path)
    prep_instance = DataPreparation(data)
    prep_instance.generate_histogram()
    prep_instance.generate_pdf()
    prep_instance.generate_pie_chart()



if __name__ == "__main__":
    main()