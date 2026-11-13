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

    preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)    
    logistic_results = logisticRegression(preprocessor, X_train, X_test, y_train, y_test)    
    ssg_results = SGDModel(preprocessor, X_train, X_test, y_train, y_test)    
    decisionTree_results, tree_pred = decisionTree(preprocessor, X_train, X_test, y_train, y_test)  
    rf_gb_results = ensembleLearning(preprocessor, X_train, X_test, y_train, y_test, tree_pred)
    randomForest_results = rf_gb_results[0]
    gradientBoosting_result = rf_gb_results[1]

    all_results = [
        logistic_results,
        ssg_results,
        decisionTree_results,
        randomForest_results,
        gradientBoosting_result
        ]
    model_names = [
        "Logistic Regression",
        "SSG Classifier",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ]
    result_df = pd.DataFrame(data=all_results, columns=["Accuracy", "Precision", "Recall", "F1"])
    result_df.insert(0, "모델", model_names)

    print("\n 모델별 성능 비교")
    print(result_df.to_string(index=False, float_format="%.4f"))
if __name__ == "__main__":
    main()