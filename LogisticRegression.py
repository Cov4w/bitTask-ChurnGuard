import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

def logisticRegression(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, C_value=1.0):
    # 모델 생성
    logistic_model = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(C=C_value, max_iter=1000, random_state=42))])
    # 모델 학습
    logistic_model.fit(X_train, y_train)
    print("로지스틱 회귀 학습 완료")

    
    # 예측
    pred = logistic_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    results = [accuracy, precision, recall, f1]
    return  results, logistic_model