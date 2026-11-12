import pandas as pd
from readCSV import readCsv
from CsvInfo import printInfo
from AnalysisAndVisualization import AAV
from DataPreprocessing import dataPreprocessing
from LogisticRegression import logisticRegression
from SGD import SGDModel
from DecisionTree import decisionTree
from EnsembleLearning import ensembleLearning
from PerformanceComparison import ModelPC
from CrossValidation import crossValidation
from GridSearchCV import gridSearchCV
def main():
    df = readCsv()

    #printInfo(df)
    #AAV(df)
    preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)
    
    logistic_pred, logistic_model = logisticRegression(preprocessor, X_train, X_test, y_train, y_test)    
    ssg_pred, sgd_model = SGDModel(preprocessor, X_train, X_test, y_train, y_test)    
    print("logistic_pred : \n", logistic_pred)
    print("ssg_pred : \n", ssg_pred)
    
    tree_pred, tree_model = decisionTree(preprocessor, X_train, X_test, y_train, y_test)
    rf_model, gb_model = ensembleLearning(preprocessor, X_train, X_test, y_train, y_test, tree_pred)

    models, result_df = ModelPC(logistic_model, sgd_model, tree_model, rf_model, gb_model, X_test, y_test)
    #crossValidation(models, X_train, y_train)
    gridSearchCV(rf_model, X_train, X_test, y_train, y_test)
if __name__ == "__main__":
    main()