import pandas as pd
from readCSV import readCsv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
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
class TestFigureRequest(BaseModel):
    logistic: Optional[float] = None
    SGD_alpha: Optional[float] = None
    dt_depth: Optional[int] = None
    rf_n_estimators: Optional[int] = None
    gb_learning_rate: Optional[float] = None

@app.get("/api/model-performance")
def model_performance():
    df = readCsv()

    preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)    
    logistic_results = logisticRegression(preprocessor, X_train, X_test, y_train, y_test)    
    sgd_results = SGDModel(preprocessor, X_train, X_test, y_train, y_test)    
    decisionTree_results, tree_pred = decisionTree(preprocessor, X_train, X_test, y_train, y_test)  
    rf_gb_results = ensembleLearning(preprocessor, X_train, X_test, y_train, y_test)
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

@app.get("/api/model-improvement")
def evaluate_improvement(params: TestFigureRequest):
    df = readCsv()    
    preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)  
    c_value = params.logistic if params.logistic is not None else 1.0
    logistic_results = logisticRegression(preprocessor, X_train, X_test, y_train, y_test, c_value)
    alpha_value = params.logistic if params.SGD_alpha is not None else 0.0001
    sgd_results = SGDModel(preprocessor, X_train, X_test, y_train, y_test, alpha_value)
    decisionTree_results, tree_pred = decisionTree(preprocessor, X_train, X_test, y_train, y_test, max_depth = params.dt_depth)
    n_est_val = params.rf_n_estimators if params.rf_n_estimators is not None else 100
    lr_val = params.gb_learning_rate if params.gb_learning_rate is not None else 0.1
    rf_gb_results = ensembleLearning(
        preprocessor, X_train, X_test, y_train, y_test, tree_pred,
        n_estimators=n_est_val, learning_rate=lr_val
    )

    return [
        format_metric("Logistic Regression", logistic_results),
        format_metric("SGD Classifier", sgd_results),
        format_metric("Decision Tree", decisionTree_results),
        format_metric("Random Forest", rf_gb_results[0]),
        format_metric("Gradient Boosting", rf_gb_results[1])
    ]
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)