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

def ensembleLearning(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, tree_pred:np.ndarray):
    rf_model = Pipeline([("preprocessor", preprocessor), 
                         ("model", RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1))])
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    """
    print("Random Forest Accuracy : ", accuracy_score(y_test, rf_pred))
    print("Random Forest Recall : ", recall_score(y_test, rf_pred))
    print("Random Forest F1 : ", f1_score(y_test, rf_pred))
    """
    #그래디언트 부스팅 모델 생성
    gb_model = Pipeline([("preprocessor", preprocessor), 
                         ("model", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))])
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    """
    print("Gradient Boosting Accuracy : ", accuracy_score(y_test, gb_pred))
    print("Gradient Boosting Recall : ", recall_score(y_test, gb_pred))
    print("Gradient Boosting F1 : ", f1_score(y_test, gb_pred))
    print("Decision Tree")
    print(classification_report(y_test, tree_pred))
    print("Random Forest")
    print(classification_report(y_test, rf_pred))
    print("Gradient Boosting")
    print(classification_report(y_test, gb_pred))
    """
    return rf_model, gb_model