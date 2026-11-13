import pandas as pd
from readCSV import readCsv
from fastapi import FastAPI
import uvicorn
from DataPreprocessing import dataPreprocessing
from LogisticRegression import logisticRegression
from SGD import SGDModel
from DecisionTree import decisionTree
from EnsembleLearning import ensembleLearning

app = FastAPI()

def format_metric(model_name:str, res:list):
    return{
        "ModelName": model_name,            
        "Accuracy": round(float(res[0]), 4), 
        "Precision": round(float(res[1]), 4),
        "Recall": round(float(res[2]), 4),
        "F1": round(float(res[3]), 4)
    }

@app.get("/api/model-performance")
def model_performance():
    df = readCsv()

    preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)    
    logistic_results = logisticRegression(preprocessor, X_train, X_test, y_train, y_test)    
    sgd_results = SGDModel(preprocessor, X_train, X_test, y_train, y_test)    
    decisionTree_results, tree_pred = decisionTree(preprocessor, X_train, X_test, y_train, y_test)  
    rf_gb_results = ensembleLearning(preprocessor, X_train, X_test, y_train, y_test, tree_pred)
    randomForest_results = rf_gb_results[0]
    gradientBoosting_results = rf_gb_results[1]   

    response_data = [
        format_metric("Logistic Regression", logistic_results),
        format_metric("SGD Classifier", sgd_results),
        format_metric("Decision Tree", decisionTree_results),
        format_metric("Random Forest", randomForest_results),
        format_metric("Gradient Boosting", gradientBoosting_results)
    ]
    return response_data
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)