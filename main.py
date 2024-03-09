from DataProcessing import read_csv_data, DataPreparation
from problem_detection import CollinearDetect
from outliers import OutlierDetection
from challenge_one import ChallengeOne
from challenge_two import ChallengeTwo

def main():
    csv_file_path = "winequality-red.csv"
    data = read_csv_data(csv_file_path)
   
    ''' temporary comment out to avoid wasting time on data profile. Remove for submission

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
    outlier_instance = OutlierDetection(data)
    outlier_instance.count_outliers_by_feature()
    outlier_instance.calculate_z_scores()

    '''

   

    c1 = ChallengeOne(data)
    c1.oversample()
    c1.inverse_prior_probabilities()
    c1.apply_smote()
    o, s, i, test = c1.test_and_print()

    print(o)
    print(s)
    print(i)
    print(test)

    
    '''
    c2 = ChallengeTwo(data)
    f = c2.feat_select()
    print('Feature selection done')
    d = c2.dimen_reduction()
    print('Dimensional Reduction Done')
    r = c2.regularization()
    print('regularization done')
    test_feature = c2.test_acc()
    
    
    print(f)
    print(d)
    print(r)
    print(test_feature)
    '''


if __name__ == "__main__":
    main()