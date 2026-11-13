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
from GridSearchCV import gridSearchCV

app = FastAPI()

saved_models = {}
saved_data = {}

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

    saved_data["X_train"], saved_data["X_test"] = X_train, X_test
    saved_data["y_train"], saved_data["y_test"] = y_train, y_test

    logistic_results, lr_model = logisticRegression(preprocessor, X_train, X_test, y_train, y_test)    
    sgd_results, sgd_model = SGDModel(preprocessor, X_train, X_test, y_train, y_test)    
    decisionTree_results, tree_pred, tree_model = decisionTree(preprocessor, X_train, X_test, y_train, y_test)  
    (rf_res, gb_res), (rf_model, gb_model) = ensembleLearning(preprocessor, X_train, X_test, y_train, y_test)
    randomForest_results = rf_res
    gradientBoosting_results = gb_res 

    saved_models["Logistic Regression"] = lr_model
    saved_models["SGD Classifier"] = sgd_model
    saved_models["Decision Tree"] = tree_model
    saved_models["Random Forest"] = rf_model
    saved_models["Gradient Boosting"] = gb_model

    response_data = [
        format_metric("Logistic Regression", logistic_results),
        format_metric("SGD Classifier", sgd_results),
        format_metric("Decision Tree", decisionTree_results),
        format_metric("Random Forest", randomForest_results),
        format_metric("Gradient Boosting", gradientBoosting_results)
    ]
    return response_data

@app.post("/api/model-improvement")
def evaluate_improvement(params: TestFigureRequest):
    df = readCsv()    
    preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)  
    c_value = params.logistic if params.logistic is not None else 1.0
    logistic_results,_ = logisticRegression(preprocessor, X_train, X_test, y_train, y_test, c_value)
    alpha_value = params.logistic if params.SGD_alpha is not None else 0.0001
    sgd_results,_ = SGDModel(preprocessor, X_train, X_test, y_train, y_test, alpha_value)
    decisionTree_results, _,_ = decisionTree(preprocessor, X_train, X_test, y_train, y_test, max_depth = params.dt_depth)
    
    n_est_val = params.rf_n_estimators if params.rf_n_estimators is not None else 100
    lr_val = params.gb_learning_rate if params.gb_learning_rate is not None else 0.1
    (rf_res, gb_res),_ = ensembleLearning(
        preprocessor, X_train, X_test, y_train, y_test, 
        n_estimators=n_est_val, learning_rate=lr_val
    )

    return [
        format_metric("Logistic Regression", logistic_results),
        format_metric("SGD Classifier", sgd_results),
        format_metric("Decision Tree", decisionTree_results),
        format_metric("Random Forest", rf_res),
        format_metric("Gradient Boosting", gb_res)
    ]

@app.get("/api/model-report")
def model_report():
    if not saved_models:
        model_performance()
    report_result = gridSearchCV(
        models_dict=saved_models,
        X_train=saved_data["X_train"],
        X_test=saved_data["X_test"],
        y_train=saved_data["y_train"],
        y_test=saved_data["y_test"]
    )

    return report_result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)