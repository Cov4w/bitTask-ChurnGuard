import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

def ensembleLearning(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, 
                     n_estimators=100, learning_rate=0.1):
    rf_model = Pipeline([("preprocessor", preprocessor), 
                         ("model", RandomForestClassifier(n_estimators=n_estimators, max_depth=7, random_state=42, n_jobs=-1))])
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)

    #그래디언트 부스팅 모델 생성
    gb_model = Pipeline([("preprocessor", preprocessor), 
                         ("model", GradientBoostingClassifier(n_estimators=100, learning_rate=learning_rate, max_depth=3, random_state=42))])
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
    gb_accuracy = accuracy_score(y_test, gb_pred)
    gb_precision = precision_score(y_test, gb_pred)
    gb_recall = recall_score(y_test, gb_pred)
    gb_f1 = f1_score(y_test, gb_pred)
    rf_results = [rf_accuracy, rf_precision, rf_recall, rf_f1]
    gb_results = [gb_accuracy, gb_precision, gb_recall, gb_f1]
    return [rf_results, gb_results],[rf_model, gb_model]