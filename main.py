import pandas as pd
from readCSV import readCsv
from CsvInfo import printInfo
from AnalysisAndVisualization import AAV
from DataPreprocessing import dataPreprocessing
from LogisticRegression import logisticRegression
from SGD import SGDModel
from DecisionTree import decisionTree

def main():
    df = readCsv()

    #printInfo(df)
    #AAV(df)
    preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)
    """
    logistic_pred = logisticRegression(preprocessor, X_train, X_test, y_train, y_test)    
    ssg_pred = SGDModel(preprocessor, X_train, X_test, y_train, y_test)    
    print("logistic_pred : \n", logistic_pred)
    print("ssg_pred : \n", ssg_pred)
    """
    decisionTree(preprocessor, X_train, X_test, y_train, y_test)
if __name__ == "__main__":
    main()